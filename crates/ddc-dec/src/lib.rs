//! ddc-dec — DEX → Java decompiler front-end over `jdc-core`.
//!
//! Pipeline per method: DEX code units → CFG → per-block register→IR
//! lifting → `jdc-core` structuring/conversion → refinement passes →
//! emission. Class-level rendering mirrors jcdc's `classdec` but reads
//! DEX metadata (no signatures, no generics).

pub mod cfg;
pub mod classdec;

pub use classdec::{sanitize_fq, ClassOptions};
pub mod ctx;
pub mod lift;
pub mod method;
pub mod passes;
pub mod platform;
mod refscan;

use jdc_core::FxHashMap as HashMap;

use ddc_dex::annotations::{self, EncodedValue};

/// A resolved static-field initializer.
#[derive(Debug, Clone)]
pub enum StaticValue {
    Int(i64),
    Float(f32),
    Double(f64),
    Str(String),
    Type(String),
    Boolean(bool),
    Null,
    /// (declaring class, field name) — enum constants.
    Field(String, String),
    Other,
}
use ddc_dex::{ClassDef, DexFile};
use jdc_core::types::{
    parse_field_descriptor, parse_method_descriptor, JavaType, MethodDescriptor,
};

// ---------------------------------------------------------------------------
// Access flags (Dalvik numbering, JVM-compatible subset).
// ---------------------------------------------------------------------------
pub mod access {
    pub const ACC_PUBLIC: u32 = 0x1;
    pub const ACC_PRIVATE: u32 = 0x2;
    pub const ACC_PROTECTED: u32 = 0x4;
    pub const ACC_STATIC: u32 = 0x8;
    pub const ACC_FINAL: u32 = 0x10;
    pub const ACC_SYNCHRONIZED: u32 = 0x20;
    pub const ACC_BRIDGE: u32 = 0x40;
    pub const ACC_VARARGS: u32 = 0x80;
    pub const ACC_NATIVE: u32 = 0x100;
    pub const ACC_INTERFACE: u32 = 0x200;
    pub const ACC_ABSTRACT: u32 = 0x400;
    pub const ACC_STRICT: u32 = 0x800;
    pub const ACC_SYNTHETIC: u32 = 0x1000;
    pub const ACC_ANNOTATION: u32 = 0x2000;
    pub const ACC_ENUM: u32 = 0x4000;
    pub const ACC_CONSTRUCTOR: u32 = 0x1_0000;
    pub const ACC_DECLARED_SYNCHRONIZED: u32 = 0x2_0000;
}

/// One field of a pooled class.
#[derive(Debug, Clone)]
pub struct PoolField {
    pub name: String,
    /// Field descriptor (`I`, `Ljava/lang/String;`, ...).
    pub desc: String,
    pub access: u32,
    pub is_static: bool,
}

/// One method of a pooled class.
#[derive(Debug, Clone)]
pub struct PoolMethod {
    /// Shared with the dex string table (the idx IS the dedup key).
    pub name: std::sync::Arc<str>,
    /// Method descriptor, shared per proto (a dex's proto table is the
    /// descriptor dedup layer — 1.9M methods on lark share ~1 proto
    /// table's worth of unique descriptors).
    pub desc: std::sync::Arc<str>,
    pub access: u32,
    pub code_off: u32,
    pub debug_info_off: u32,
    /// Which DEX image the body lives in.
    pub dex_idx: usize,
}

impl PoolMethod {
    pub fn parsed_desc(&self) -> Option<MethodDescriptor> {
        parse_method_descriptor(&self.desc)
    }
    pub fn is_static(&self) -> bool {
        self.access & access::ACC_STATIC != 0
    }
    pub fn is_abstract_or_native(&self) -> bool {
        self.access & (access::ACC_ABSTRACT | access::ACC_NATIVE) != 0
    }
}

/// One pool entry: either already built, or name-registered with its
/// (dex index, class_def index) locator for on-demand construction.
enum ClassEntry {
    Eager(PoolClass),
    Lazy {
        at: (usize, usize),
        pc: std::sync::OnceLock<PoolClass>,
    },
}

/// A class materialized from one `class_def_item` (with ids resolved to
/// names and the class data expanded).
#[derive(Debug, Clone)]
pub struct PoolClass {
    pub name: String,
    pub access: u32,
    /// `None` for `java/lang/Object` roots.
    pub super_name: Option<String>,
    pub interfaces: Vec<String>,
    pub source_file: Option<String>,
    pub static_fields: Vec<PoolField>,
    pub instance_fields: Vec<PoolField>,
    pub direct_methods: Vec<PoolMethod>,
    pub virtual_methods: Vec<PoolMethod>,
    /// Static initial values aligned with the head of `static_fields`.
    pub static_values: Vec<StaticValue>,
    /// Nesting evidence with annotation ids resolved to names.
    pub nesting: ResolvedNesting,
    /// Which DEX image the class body lives in (provenance header).
    pub dex_idx: usize,
}

/// Nesting evidence, names resolved at pooling time.
#[derive(Debug, Clone, Default)]
pub struct ResolvedNesting {
    pub enclosing_class: Option<String>,
    /// (class internal name, method name) of the enclosing method.
    pub enclosing_method: Option<(String, String)>,
    pub member_classes: Vec<String>,
}

impl PoolClass {
    pub fn is_interface(&self) -> bool {
        self.access & access::ACC_INTERFACE != 0
    }

    pub fn is_enum(&self) -> bool {
        self.access & access::ACC_ENUM != 0
    }

    pub fn is_synthetic(&self) -> bool {
        self.access & access::ACC_SYNTHETIC != 0
    }

    pub fn is_static_nested(&self) -> bool {
        self.access & access::ACC_STATIC != 0
    }

    /// All methods in declaration order.
    pub fn all_methods(&self) -> impl Iterator<Item = &PoolMethod> {
        self.direct_methods
            .iter()
            .chain(self.virtual_methods.iter())
    }

    /// Find a method by name + descriptor.
    pub fn find_method(&self, name: &str, desc: &str) -> Option<&PoolMethod> {
        self.all_methods()
            .find(|m| &*m.name == name && &*m.desc == desc)
    }

    /// Constructors `<init>` matching `arity` descriptor arguments.
    pub fn ctors_by_arity(&self, arity: usize) -> Vec<&PoolMethod> {
        self.all_methods()
            .filter(|m| &*m.name == "<init>")
            .filter(|m| {
                m.parsed_desc()
                    .map(|d| d.args.len() == arity)
                    .unwrap_or(false)
            })
            .collect()
    }

    pub fn field_flags_of(&self, name: &str) -> Option<u32> {
        self.static_fields
            .iter()
            .chain(self.instance_fields.iter())
            .find(|f| f.name == name)
            .map(|f| f.access)
    }
}

/// Multi-DEX class pool: classes from all images, first definition wins.
pub struct DexPool {
    /// Images as Arcs: `dex()` hands out snapshots whose borrows stay
    /// valid for the Arc's lifetime (decompile workers hold them), while
    /// `release_images` empties the image data through exclusive access
    /// when all of an image's classes are done (full-decompile driver
    /// only — progressive callers keep images whole).
    dexes: Vec<std::sync::Arc<DexFile>>,
    /// Human label per image ("weibo!classes.dex") for the provenance
    /// header; defaults to "dex N" until the CLI names the inputs.
    pub dex_labels: Vec<String>,
    /// Name → class entry. Eager pools materialize everything at add
    /// time (full decompile, tests); lazy pools register names only and
    /// materialize on first `get` (progressive `getclass` — one class's
    /// PoolClass costs annotation reads, 145k of them cost ~0.2s).
    classes: HashMap<String, ClassEntry>,
    /// Class names in insertion order (stable for output).
    pub order: Vec<String>,
    /// Per-image remaining class count once `arm_retirement` fires.
    retire_counts: std::sync::Mutex<Vec<u64>>,
    retire_armed: std::sync::atomic::AtomicBool,
    /// name → outer (computed once; `$` heuristic + dalvik annotations).
    outers: std::sync::OnceLock<HashMap<String, Option<String>>>,
    /// outer → direct children (computed once).
    children: std::sync::OnceLock<HashMap<String, Vec<String>>>,
    /// Per-image hot-reference interning (parallel to `dexes`).
    ref_caches: Vec<DexRefCache>,
    /// Synthetic-static accessor code snapshots keyed by (dex_idx,
    /// code_off). `inline_accessors` reads accessor bodies ACROSS images
    /// during worker runs; image retirement can release those bytes
    /// mid-run, making the inline decision — and with it var numbering
    /// downstream — dependent on worker completion interleaving
    /// (observed: reqable enum-constant args flipping `var0`/`var1`
    /// between identical runs; DDC_NORETIRE=1 stable). The full-decompile
    /// driver fills this BEFORE `arm_retirement`; progressive/lazy pools
    /// never retire and fall through to live reads.
    accessor_code: std::sync::OnceLock<AccessorSnapshots>,
    /// Cached package → direct-class simple names (import-collision gate).
    pkg_simples: std::sync::OnceLock<jdc_core::FxHashMap<String, jdc_core::FxHashSet<String>>>,
    /// Kotlin multi-file facade parts → public facade (see
    /// `kotlin_facade_map`).
    kotlin_facades: std::sync::OnceLock<jdc_core::FxHashMap<String, String>>,
}

/// Raw code_item bytes of every synthetic-static accessor, keyed by
/// (dex_idx, code_off) — see `snapshot_accessor_code`.
type AccessorSnapshots = HashMap<(usize, u32), std::sync::Arc<[u8]>>;

/// Lazily-filled, thread-shared reference interning for one image.
///
/// The lifter used to rebuild class names, field/method names and WHOLE
/// method descriptors per INSTRUCTION (`method_ref` alone: N param
/// Strings → `format!` descriptor → re-parse — 6-10 allocations per
/// invoke). Every entry here is built at most once per (image, table
/// index); lifts and the endless IR clones downstream are refcount
/// bumps. Slots are `OnceLock` so worker threads fill them race-free
/// without a lock on the hot path.
pub(crate) struct DexRefCache {
    proto_descs: Box<[std::sync::OnceLock<std::sync::Arc<jdc_core::types::MethodDescriptor>>]>,
    type_names: Box<[std::sync::OnceLock<std::sync::Arc<str>>]>,
    type_javas: Box<[std::sync::OnceLock<std::sync::Arc<jdc_core::types::JavaType>>]>,
}

impl DexRefCache {
    fn new(protos: usize, types: usize) -> Self {
        DexRefCache {
            proto_descs: (0..protos).map(|_| std::sync::OnceLock::new()).collect(),
            type_names: (0..types).map(|_| std::sync::OnceLock::new()).collect(),
            type_javas: (0..types).map(|_| std::sync::OnceLock::new()).collect(),
        }
    }
}

impl DexPool {
    pub fn new() -> Self {
        DexPool {
            dexes: Vec::new(),
            retire_armed: std::sync::atomic::AtomicBool::new(false),
            dex_labels: Vec::new(),
            classes: HashMap::default(),
            order: Vec::new(),
            retire_counts: std::sync::Mutex::new(Vec::new()),
            outers: std::sync::OnceLock::new(),
            children: std::sync::OnceLock::new(),
            ref_caches: Vec::new(),
            accessor_code: std::sync::OnceLock::new(),
            pkg_simples: std::sync::OnceLock::new(),
            kotlin_facades: std::sync::OnceLock::new(),
        }
    }

    /// Shared `MethodDescriptor` of a proto (one parse per image).
    pub fn proto_desc_parsed(
        &self,
        di: usize,
        proto_idx: u32,
    ) -> std::sync::Arc<jdc_core::types::MethodDescriptor> {
        let Some(dex) = self.dexes.get(di) else {
            return std::sync::Arc::new(jdc_core::types::MethodDescriptor {
                args: Vec::new(),
                ret: jdc_core::types::JavaType::Void,
            });
        };
        let build = || {
            std::sync::Arc::new(parse_proto_desc(dex, proto_idx))
        };
        match self
            .ref_caches
            .get(di)
            .and_then(|c| c.proto_descs.get(proto_idx as usize))
        {
            Some(slot) => slot.get_or_init(build).clone(),
            None => build(),
        }
    }

    /// Shared internal class name of a type id (one strip+alloc per image).
    pub fn type_name_arc(&self, di: usize, type_idx: u32) -> std::sync::Arc<str> {
        let Some(dex) = self.dexes.get(di) else {
            return std::sync::Arc::from("");
        };
        let build = || std::sync::Arc::from(dex.class_name(type_idx).as_str());
        match self
            .ref_caches
            .get(di)
            .and_then(|c| c.type_names.get(type_idx as usize))
        {
            Some(slot) => slot.get_or_init(build).clone(),
            None => build(),
        }
    }

    /// Shared parsed type of a type id.
    pub fn type_java(&self, di: usize, type_idx: u32) -> std::sync::Arc<jdc_core::types::JavaType> {
        let Some(dex) = self.dexes.get(di) else {
            return std::sync::Arc::new(jdc_core::types::JavaType::Object(
                std::sync::Arc::from("java/lang/Object"),
            ));
        };
        let build = || std::sync::Arc::new(desc_type(dex.type_name(type_idx)));
        match self
            .ref_caches
            .get(di)
            .and_then(|c| c.type_javas.get(type_idx as usize))
        {
            Some(slot) => slot.get_or_init(build).clone(),
            None => build(),
        }
    }

    /// name → outer map, computed lazily once.
    fn outer_map(&self) -> &HashMap<String, Option<String>> {
        self.outers.get_or_init(|| {
            self.order
                .iter()
                .map(|n| (n.clone(), find_outer_name(self, n)))
                .collect()
        })
    }

    /// The outer class of `name`, from the cached map (borrowed).
    pub fn outer_of(&self, name: &str) -> Option<&str> {
        self.outer_map().get(name).and_then(|o| o.as_deref())
    }

    /// Direct nested children of `internal` (any `$` depth 1), cached.
    /// (Perf: returns a borrowed slice — the owned-Vec clone ran once per
    /// class on 98k-class runs.)
    pub fn children_of(&self, internal: &str) -> &[String] {
        if self.children.get().is_none() {
            // Build FIRST, set ONCE: the previous shape set an empty map
            // to claim the OnceLock and then silently failed to store the
            // real index (`let _ = set(...)` on an initialized lock) —
            // children_of returned [] forever, so member nested classes
            // were neither inlined nor emitted as files (Guard$Report
            // vanished whole; every corpus run since the borrow refactor
            // dropped them, invisible to the syntax-only gate).
            let mut idx: HashMap<String, Vec<String>> = HashMap::default();
            for name in &self.order {
                if let Some(outer) = self.outer_of(name).map(str::to_string) {
                    idx.entry(outer).or_default().push(name.clone());
                }
            }
            // A racing thread may have set an identical map first — the
            // index is a pure function of `order`, so either copy is right.
            let _ = self.children.set(idx);
        }
        self.children
            .get()
            .and_then(|m| m.get(internal))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Adds one image; duplicate class names keep their first definition.
    /// Returns the image's index — call `set_dex_label` with it to give the
    /// provenance header a real origin instead of the default "dex N".
    pub fn add_dex(&mut self, dex: DexFile) -> usize {
        let dex_idx = self.dexes.len();
        // Pre-size the name map: 240k-entry growth rehashed repeatedly
        // (reserve_rehash showed up in corpus profiles).
        self.classes.reserve(dex.class_defs.len());
        // The annotation reader borrows the image; pool classes are built
        // before ownership moves into `self.dexes` (no full-image copy).
        for cd in &dex.class_defs {
            let name = dex.class_name(cd.class_idx);
            if self.classes.contains_key(&name) {
                continue;
            }
            let pc = pool_class_of(&dex, dex.raw(), cd, dex_idx);
            self.classes.insert(name.clone(), ClassEntry::Eager(pc));
            self.order.push(name);
        }
        self.ref_caches
            .push(DexRefCache::new(dex.proto_count(), dex.type_count()));
        self.dexes.push(std::sync::Arc::new(dex));
        self.retire_counts.lock().unwrap().push(0);
        self.dex_labels.push(format!("dex {}", dex_idx));
        dex_idx
    }

    /// Registers class NAMES only; the PoolClass (annotation reads
    /// included) is materialized on first `get`. Duplicate class names
    /// keep their first definition, matching `add_dex`.
    pub fn add_dex_lazy(&mut self, dex: DexFile) -> usize {
        let dex_idx = self.dexes.len();
        self.classes.reserve(dex.class_defs.len());
        for (ci, cd) in dex.class_defs.iter().enumerate() {
            let name = dex.class_name(cd.class_idx);
            if self.classes.contains_key(&name) {
                continue;
            }
            self.classes.insert(
                name.clone(),
                ClassEntry::Lazy {
                    at: (dex_idx, ci),
                    pc: std::sync::OnceLock::new(),
                },
            );
            self.order.push(name);
        }
        self.ref_caches
            .push(DexRefCache::new(dex.proto_count(), dex.type_count()));
        self.dexes.push(std::sync::Arc::new(dex));
        self.retire_counts.lock().unwrap().push(0);
        self.dex_labels.push(format!("dex {}", dex_idx));
        dex_idx
    }

    /// Label the image at `idx` (see `add_dex`).
    pub fn set_dex_label(&mut self, idx: usize, label: impl Into<String>) {
        if let Some(slot) = self.dex_labels.get_mut(idx) {
            *slot = label.into();
        }
    }

    pub fn len(&self) -> usize {
        self.classes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    pub fn get(&self, internal: &str) -> Option<&PoolClass> {
        self.get_inner(internal, None)
    }

    /// `get` with optional pre-taken raw-image snapshots (one per
    /// image). `materialize_all` hoists the `DexFile::raw()` lock out of
    /// the per-class loop — the parallel materialize used to take the
    /// same per-image lock once PER CLASS across all threads.
    fn get_inner(&self, internal: &str, raws: Option<&[&[u8]]>) -> Option<&PoolClass> {
        match self.classes.get(internal)? {
            ClassEntry::Eager(pc) => Some(pc),
            ClassEntry::Lazy { at, pc } => {
                if let Some(built) = pc.get() {
                    return Some(built);
                }
                let (di, ci) = *at;
                let dex = self.dexes.get(di)?;
                let cd = dex.class_defs.get(ci)?;
                let raw = match raws {
                    Some(r) => r.get(di).copied().unwrap_or(&[]),
                    None => dex.raw(),
                };
                let built = pool_class_of(dex, raw, cd, di);
                Some(pc.get_or_init(move || built))
            }
        }
    }

    /// Materialize every lazy entry (parallel across the given chunk
    /// count). After this the pool is observationally identical to an
    /// eagerly-built one — including the annotation-aware outer-map —
    /// while the annotation reads ran on worker threads instead of the
    /// serial pool build (~0.2s on weibo).
    pub fn materialize_all(&self, threads: usize) {
        let keys: Vec<String> = self
            .classes
            .iter()
            .filter(|(_, e)| matches!(e, ClassEntry::Lazy { .. }))
            .map(|(k, _)| k.clone())
            .collect();
        if keys.is_empty() {
            return;
        }
        // One raw() snapshot per image for the whole sweep (see
        // get_inner). Safe window: materialize_all completes before
        // retirement is armed, so no image releases while these slices
        // are alive.
        let raws: Vec<&[u8]> = self.dexes.iter().map(|d| d.raw()).collect();
        let threads = threads.max(1);
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(k) = keys.get(i) else { break };
                    let _ = self.get_inner(k, Some(&raws));
                });
            }
        });
    }

    /// `get` without materializing: None for not-yet-built lazy entries.
    fn get_if_materialized(&self, internal: &str) -> Option<&PoolClass> {
        match self.classes.get(internal)? {
            ClassEntry::Eager(pc) => Some(pc),
            ClassEntry::Lazy { pc, .. } => pc.get(),
        }
    }

    /// Name registered (no materialization).
    pub fn has_name(&self, internal: &str) -> bool {
        self.classes.contains_key(internal)
    }

    /// One image as an Arc snapshot (borrows of the snapshot live as
    /// long as the caller holds the Arc).
    pub fn dex(&self, idx: usize) -> Option<std::sync::Arc<DexFile>> {
        self.dexes.get(idx).cloned()
    }

    pub fn dex_count(&self) -> usize {
        self.dexes.len()
    }

    /// Package → simple names of its direct classes, cached once per
    /// pool (the import-collision gate consults it per class — the
    /// uncached scan was O(classes²) and 10×'d weixin's wall time).
    pub fn package_simples(&self) -> &jdc_core::FxHashMap<String, jdc_core::FxHashSet<String>> {
        self.pkg_simples.get_or_init(|| {
            let mut m: jdc_core::FxHashMap<String, jdc_core::FxHashSet<String>> =
                jdc_core::FxHashMap::default();
            for n in &self.order {
                if let Some((pkg, simple)) = n.rsplit_once('/') {
                    m.entry(pkg.to_string())
                        .or_default()
                        .insert(simple.to_string());
                }
            }
            m
        })
    }

    /// Kotlin multi-file class facades: the compiler splits a file facade
    /// (`kotlin.text.StringsKt`) into package-private parts
    /// (`StringsKt__StringsKt`, `StringsKt__StringsJVMKt`) and call sites
    /// target the PARTS directly. Java cannot access a package-private
    /// class cross-package ("StringsKt__StringsKt在kotlin.text中不是公共的"
    /// — 1.4k lark root errors), but the public facade EXTENDS the parts,
    /// so static members resolve through it: rewrite the reference's owner
    /// class to the facade. part → base, gated on base existing, being
    /// public, and transitively extending the part (static inheritance is
    /// the resolution guarantee).
    pub fn kotlin_facade_map(&self) -> &jdc_core::FxHashMap<String, String> {
        self.kotlin_facades.get_or_init(|| {
            let mut m: jdc_core::FxHashMap<String, String> = jdc_core::FxHashMap::default();
            for name in &self.order {
                let Some(i) = name.find("__") else { continue };
                let base = &name[..i];
                if base.is_empty() || self.get(base).is_none() {
                    continue;
                }
                let pub_base = self
                    .get(base)
                    .is_some_and(|c| c.access & 0x1 != 0);
                if !pub_base {
                    continue;
                }
                // base must transitively extend the part.
                let mut cur = self.get(base).and_then(|c| c.super_name.clone());
                let mut hops = 0u32;
                let mut reaches = false;
                while let Some(c) = cur {
                    if c == *name {
                        reaches = true;
                        break;
                    }
                    if hops >= 8 {
                        break;
                    }
                    hops += 1;
                    cur = self.get(&c).and_then(|cc| cc.super_name.clone());
                }
                if reaches {
                    m.insert(name.clone(), base.to_string());
                }
            }
            m
        })
    }

    pub fn class_names(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(|s| s.as_str())
    }

    /// True when `sub` is assignable to `sup` (internal names), walking the
    /// pool's hierarchy. Unknown classes are never assignable (conservative).
    pub fn is_subtype(&self, sub: &str, sup: &str) -> bool {
        if sub == sup {
            return true;
        }
        // Arrays: covariance by element type when both are arrays.
        if let (Some(sub_elem), Some(sup_elem)) = (array_elem(sub), array_elem(sup)) {
            if sub_elem.starts_with('L')
                && sup_elem.starts_with('L')
                && sub_elem.ends_with(';')
                && sup_elem.ends_with(';')
            {
                return self.is_subtype(
                    &sub_elem[1..sub_elem.len() - 1],
                    &sup_elem[1..sup_elem.len() - 1],
                );
            }
            return sub_elem == sup_elem;
        }
        let mut cur = self.get(sub);
        let mut hops = 0;
        while let Some(c) = cur {
            hops += 1;
            if hops > 64 {
                return false;
            }
            for i in &c.interfaces {
                if i == sup || self.is_subtype(i, sup) {
                    return true;
                }
            }
            match &c.super_name {
                Some(s) if s == sup => return true,
                Some(s) if s != "java/lang/Object" => {
                    cur = self.get(s);
                }
                _ => return false,
            }
        }
        false
    }
}

impl Default for DexPool {
    fn default() -> Self {
        Self::new()
    }
}

fn array_elem(desc: &str) -> Option<&str> {
    desc.strip_prefix('[')
}

fn resolve_static_value(v: &EncodedValue, dex: &DexFile) -> StaticValue {
    match v {
        EncodedValue::Byte(x) => StaticValue::Int(*x as i64),
        EncodedValue::Short(x) => StaticValue::Int(*x as i64),
        EncodedValue::Char(x) => StaticValue::Int(*x as i64),
        EncodedValue::Int(x) => StaticValue::Int(*x as i64),
        EncodedValue::Long(x) => StaticValue::Int(*x),
        EncodedValue::Float(x) => StaticValue::Float(*x),
        EncodedValue::Double(x) => StaticValue::Double(*x),
        EncodedValue::String(i) => StaticValue::Str(dex.string(*i).to_string()),
        EncodedValue::Type(i) => StaticValue::Type(dex.class_name(*i)),
        EncodedValue::Boolean(b) => StaticValue::Boolean(*b),
        EncodedValue::Null => StaticValue::Null,
        EncodedValue::Enum(i) | EncodedValue::Field(i) => {
            let f = dex.field(*i);
            StaticValue::Field(
                dex.class_name(f.class_idx),
                dex.string(f.name_idx).to_string(),
            )
        }
        _ => StaticValue::Other,
    }
}

fn pool_class_of(dex: &DexFile, raw: &[u8], cd: &ClassDef, dex_idx: usize) -> PoolClass {
    let name = dex.class_name(cd.class_idx);
    let super_name = if cd.superclass_idx == ddc_dex::NO_INDEX {
        None
    } else {
        let s = dex.class_name(cd.superclass_idx);
        if s.is_empty() || s == "java/lang/Object" {
            None
        } else {
            Some(s)
        }
    };
    let interfaces = dex
        .interfaces_of(cd)
        .into_iter()
        .map(|t| dex.class_name(t))
        .collect();
    let source_file = if cd.source_file_idx == ddc_dex::NO_INDEX {
        None
    } else {
        Some(dex.string(cd.source_file_idx).to_string())
    };

    let data = dex.class_data(cd);
    let mk_field = |ef: &ddc_dex::EncodedField, is_static: bool| {
        let f = dex.field(ef.field_idx);
        PoolField {
            name: dex.string(f.name_idx).to_string(),
            desc: dex.type_name(f.type_idx).to_string(),
            access: ef.access_flags,
            is_static,
        }
    };
    let static_fields: Vec<PoolField> = data
        .static_fields
        .iter()
        .map(|f| mk_field(f, true))
        .collect();
    let instance_fields: Vec<PoolField> = data
        .instance_fields
        .iter()
        .map(|f| mk_field(f, false))
        .collect();
    let mk_method = |em: &ddc_dex::EncodedMethod| {
        let m = dex.method(em.method_idx);
        let (code_off, debug_info_off) = match dex.debug_info_off_at(em.code_off) {
            Some(d) => (em.code_off, d),
            None => (0, 0),
        };
        PoolMethod {
            name: dex.string_arc(m.name_idx),
            desc: dex.proto_desc(m.proto_idx),
            access: em.access_flags,
            code_off,
            debug_info_off,
            dex_idx,
        }
    };
    let direct_methods: Vec<PoolMethod> = data.direct_methods.iter().map(&mk_method).collect();
    let virtual_methods: Vec<PoolMethod> = data.virtual_methods.iter().map(&mk_method).collect();

    let static_values: Vec<StaticValue> = dex
        .static_values(cd.static_values_off)
        .into_iter()
        .map(|v| resolve_static_value(&v, dex))
        .collect();
    let anns = if cd.annotations_off != 0 {
        annotations::read_class_annotations(raw, cd.annotations_off)
    } else {
        Vec::new()
    };
    let raw = annotations::nesting_from(&anns, &|t: u32| dex.type_name(t).to_string());
    let nesting = ResolvedNesting {
        enclosing_class: raw.enclosing_class.map(|t| dex.class_name(t)),
        enclosing_method: raw.enclosing_method.map(|m| {
            let mid = dex.method(m);
            (
                dex.class_name(mid.class_idx),
                dex.string(mid.name_idx).to_string(),
            )
        }),
        member_classes: raw
            .member_classes
            .into_iter()
            .map(|t| dex.class_name(t))
            .collect(),
    };

    PoolClass {
        name,
        access: cd.access_flags,
        super_name,
        interfaces,
        source_file,
        static_fields,
        instance_fields,
        direct_methods,
        virtual_methods,
        static_values,
        nesting,
        dex_idx,
    }
}

/// JavaType for a field/method descriptor segment.
/// Parse a proto's descriptor into a MethodDescriptor straight from the
/// tables (no string round-trip).
fn parse_proto_desc(
    dex: &ddc_dex::DexFile,
    proto_idx: u32,
) -> jdc_core::types::MethodDescriptor {
    let proto = dex.proto(proto_idx);
    let args = dex
        .proto_params(proto_idx)
        .iter()
        .map(|&t| desc_type(dex.type_name(t)))
        .collect();
    jdc_core::types::MethodDescriptor {
        args,
        ret: desc_type(dex.type_name(proto.return_type_idx)),
    }
}

pub fn desc_type(desc: &str) -> JavaType {
    parse_field_descriptor(desc).unwrap_or(JavaType::Object("java/lang/Object".into()))
}

/// Nesting evidence for `internal`: dalvik annotations first, then the
/// `$`-name heuristic against the pool.
pub fn find_outer_name(pool: &DexPool, internal: &str) -> Option<String> {
    // Annotation refinement only for ALREADY-MATERIALIZED classes: eager
    // pools (full decompile) behave exactly as before; lazy pools
    // (progressive getclass) fall straight to the `$` chain without
    // materializing the world.
    if let Some(pc) = pool.get_if_materialized(internal) {
        if let Some(enc) = &pc.nesting.enclosing_class {
            return Some(enc.clone());
        }
    }
    let mut rest = internal;
    while let Some(d) = rest.rfind('$') {
        let cand = &rest[..d];
        if pool.has_name(cand) {
            return Some(cand.to_string());
        }
        rest = cand;
    }
    None
}

/// True when the `$` tail names a plain member class (not anonymous/local/
/// lambda), i.e. the class renders inside its outer's compilation unit.
fn clean_member_tail(rest: &str) -> bool {
    if rest.starts_with('-') {
        return false;
    }
    let tail = rest.rsplit('$').next().unwrap_or(rest);
    if tail.is_empty() || tail.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    true
}

/// Classes to emit as their own compilation units: top-level classes plus
/// anonymous/local/lambda-shaped ones (clean members render inline).
impl DexPool {
    /// Arm image retirement (full-decompile driver only): count one
    /// pending class per image by pool ownership.
    /// Snapshot every synthetic-static accessor body (the exact predicate
    /// `inline_accessors` uses) as RAW code_item bytes, so cross-image
    /// accessor reads during worker runs survive image retirement.
    /// Call AFTER materialization and BEFORE `arm_retirement`; idempotent
    /// (first fill wins). Raw memcpy, no decode — accessors are numerous
    /// (weixin ~100k) but only a fraction are ever inlined, so decoding
    /// stays lazy at the use site. The slice bound covers the fixed
    /// header + insns + try items with generous handler slack (accessor
    /// tries are vanishingly rare; the snapshot is a deterministic
    /// function of the image either way).
    pub fn snapshot_accessor_code(&self) {
        let mut snap: AccessorSnapshots = HashMap::default();
        let mut bytes_total = 0usize;
        for name in &self.order {
            let Some(pc) = self.get_if_materialized(name) else {
                continue;
            };
            for m in pc.direct_methods.iter().chain(pc.virtual_methods.iter()) {
                if m.code_off == 0
                    || !m.is_static()
                    || m.access & crate::access::ACC_SYNTHETIC == 0
                {
                    continue;
                }
                if snap.contains_key(&(m.dex_idx, m.code_off)) {
                    continue;
                }
                let Some(dex) = self.dex(m.dex_idx) else {
                    continue;
                };
                let raw = dex.raw();
                let off = m.code_off as usize;
                if off + 16 > raw.len() {
                    continue;
                }
                let u2 = |o: usize| u16::from_le_bytes([raw[o], raw[o + 1]]) as usize;
                let u4 = |o: usize| {
                    u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]) as usize
                };
                let tries = u2(off + 6);
                let insns = u4(off + 12);
                let mut end = off + 16 + 2 * insns;
                if tries > 0 {
                    // pad to 4-align + try_item[tries] + handler blobs
                    // (uleb-coded; slack covers realistic accessor shapes).
                    end = end.next_multiple_of(4) + 8 * tries + 256 + 64 * tries;
                }
                let end = end.min(raw.len());
                let slice: std::sync::Arc<[u8]> = raw[off..end].into();
                bytes_total += slice.len();
                snap.insert((m.dex_idx, m.code_off), slice);
            }
        }
        if std::env::var("DDC_WALL").is_ok() {
            eprintln!(
                "[wall] accessor snapshot: {} entries, {} KB",
                snap.len(),
                bytes_total / 1024
            );
        }
        let _ = self.accessor_code.set(snap);
    }

    /// Snapshotted accessor code_item bytes, when the snapshot was taken.
    /// Parse with `CodeItem::parse(&bytes, 0)`.
    pub fn accessor_code(&self, dex_idx: usize, code_off: u32) -> Option<std::sync::Arc<[u8]>> {
        self.accessor_code.get()?.get(&(dex_idx, code_off)).cloned()
    }

    pub fn arm_retirement(&self) {
        self.retire_armed
            .store(true, std::sync::atomic::Ordering::Release);
        let mut counts = self.retire_counts.lock().unwrap();
        for c in counts.iter_mut() {
            *c = 0;
        }
        for entry in self.classes.values() {
            let di = match entry {
                ClassEntry::Eager(pc) => pc.dex_idx,
                ClassEntry::Lazy { at, .. } => at.0,
            };
            if let Some(c) = counts.get_mut(di) {
                *c += 1;
            }
        }
    }

    /// Report one class finished; Some(image) when it was the image's
    /// last class (driver batches these into release_images).
    pub fn report_class_done(&self, class_name: &str) -> Option<usize> {
        if !self.retire_armed.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        let entry = self.classes.get(class_name)?;
        let di = match entry {
            ClassEntry::Eager(pc) => pc.dex_idx,
            ClassEntry::Lazy { at, .. } => at.0,
        };
        let mut counts = self.retire_counts.lock().unwrap();
        match counts.get_mut(di) {
            Some(c) => {
                *c = c.saturating_sub(1);
                if *c == 0 {
                    Some(di)
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Release the images' inflated bytes. Safe: the driver calls this
    /// only when every class of each image is emitted or failed; new
    /// `dex()` snapshots still work (tables valid, code accessors empty).
    pub fn release_images(&self, indexes: &[usize]) {
        let mut counts = self.retire_counts.lock().unwrap();
        for &i in indexes {
            if let Some(dex) = self.dexes.get(i) {
                // Mark the shared image: code accessors go empty
                // immediately; the bytes drop when the last snapshot
                // drops (workers hold snapshots only mid-class).
                dex.mark_released();
            }
            if let Some(c) = counts.get_mut(i) {
                *c = u64::MAX; // released marker
            }
        }
    }
}

pub fn top_level_classes(pool: &DexPool) -> Vec<String> {
    pool.class_names()
        .filter(|name| match pool.outer_of(name) {
            None => true,
            Some(outer) => {
                if pool.get(outer).is_none() {
                    return true;
                }
                // The outer can come from an ANNOTATION (EnclosingClass)
                // with no naming relationship to this class — obfuscated
                // apps pair a 1-char name with a long enclosing descriptor
                // (weixin), which made the old blind slice panic. Only a
                // real `outer$tail` prefix yields a member tail; anything
                // else is a standalone unit.
                let rest = name
                    .strip_prefix(outer)
                    .and_then(|t| t.strip_prefix('$'))
                    .unwrap_or("");
                !clean_member_tail(rest)
            }
        })
        .map(|n| n.to_string())
        .collect()
}

/// Sanitized form of one internal-name segment — the shared mapping the
/// declaration site (`print_class_name` → `sanitize_ref`), the CLASS
/// lookup and the CLI writer must all agree on. Exported so the writer
/// cannot drift from the declaration (see `sanitize_fq`).
pub fn sanitize_seg(seg: &str) -> String {
    classdec::sanitize_fq(seg)
}

/// Sanitized internal PATH form of a class name: the exact on-disk
/// identity `source_path` writes and the declaration claims.
pub fn sanitize_internal(internal: &str) -> String {
    internal
        .split('/')
        .map(classdec::sanitize_fq)
        .collect::<Vec<_>>()
        .join("/")
}

/// Deterministic case-collision renames over the FILE-emission set:
/// classes whose internal names differ only in letter case cannot share
/// one case-insensitive directory; the first (sorted) member of each
/// group keeps its name, the others gain `_2`, `_3`, … on the simple
/// segment. The map carries identity entries for unrenamed file-level
/// classes (they anchor nested prefix walks in apply_class_rename).
pub fn case_rename_map(pool: &DexPool) -> HashMap<String, String> {
    use std::collections::HashMap;
    let mut groups: HashMap<String, Vec<String>> = HashMap::default();
    for t in top_level_classes(pool) {
        groups.entry(t.to_lowercase()).or_default().push(t);
    }
    let folds: std::collections::HashSet<String> = groups.keys().cloned().collect();
    let mut map = HashMap::default();
    for (_, mut members) in groups {
        members.sort();
        for (i, m) in members.iter().enumerate() {
            if i == 0 {
                map.insert(m.clone(), m.clone());
                continue;
            }
            // Suffix the simple segment; keep incrementing if the fold
            // of the suffixed name is already taken by a real class.
            let cut = m.rfind('/').map(|x| x + 1).unwrap_or(0);
            let mut n = i + 1;
            loop {
                let cand = format!("{}{}_{}", &m[..cut], &m[cut..], n);
                let fold = cand.to_lowercase();
                if !folds.contains(&fold) {
                    map.insert(m.clone(), cand);
                    break;
                }
                n += 1;
            }
        }
    }
    map
}

/// Display-level renames for two nested-class shapes that render
/// uncompilable or inconsistent output:
///
/// 1. A member class whose simple tail equals its rendered parent's
///    tail (`x0$a$a` inside `x0$a`): javac rejects a member class with
///    the same simple name as its immediately enclosing class ("已在类
///    x0中定义了类 x0.a") — a shape Kotlin lambda families hit by the
///    thousand (lark alone: 22k). Bump the tail: `a` → `a2`, `a3`, …
/// 2. An orphaned intermediate (`x0$a$b` with `x0$a` absent from the
///    pool): the member DECLARES as `b` inside `x0` but references
///    printed `x0.a$b` — a type that exists nowhere. Flatten the
///    display name to `x0$b` so every site agrees.
///
/// Both are pure display renames keyed by internal name; declarations,
/// ctor names, file names and every type reference funnel through
/// apply_class_rename, so one map keeps all sites consistent. Children
/// Display-level renames for nested member classes that render
/// uncompilable or inconsistent output:
///
/// 1. A member whose simple tail equals ANY enclosing class's simple
///    name in its nesting chain. javac rejects more than the immediate
///    parent (`class a { static class a2 { static class a {} } }` —
///    member `a` two levels under `a` — is "已在类 X中定义了类 X.a");
///    Kotlin lambda families (`x0$a$a` inside `x0$a`) hit it by the
///    thousand (lark alone: 22k). Bump the tail until it clears every
///    ancestor name: `a` → `a3`.
/// 2. An orphaned intermediate (`x0$a$b` with `x0$a` absent): the
///    member DECLARES as `b` inside `x0` but references printed
///    `x0.a$b` — a type that exists nowhere. Flatten the display to
///    `x0$b` so every site agrees.
///
/// Candidates build on the parent's DISPLAY name (ancestors are
/// renamed first — ascending `$`-depth order), so declarations,
/// constructor names, file names and every type reference stay
/// consistent through apply_class_rename. All are pure display
/// renames keyed by internal name.
fn nested_collision_renames(
    pool: &DexPool,
    map: &mut HashMap<String, String>,
    fam_segs: &SegMap,
) {
    // Ancestors before descendants: a parent's rename must be in the
    // map when its children compute their display chains.
    let mut names: Vec<&String> = pool.order.iter().collect();
    names.sort_by_key(|n| n.matches('$').count());
    let mut assigned: jdc_core::FxHashSet<String> = jdc_core::FxHashSet::default();
    // Top-level package segments. A nested member type DISPLAYED as `a2`
    // shadows package `a2` for every qualified `a2.x` reference in its
    // file (member types outrank package names in class scope), so a
    // renamed tail must not become a package shadow — round 47's failure
    // mode at nested level: candidate `a2` (for j2/g$a) broke the SAME
    // file's `extends a2.a` ("找不到符号 类 a" + a 9450-error Object
    // cascade as every dependent of the corrupt head failed too).
    let mut pkg_segments: jdc_core::FxHashSet<String> = jdc_core::FxHashSet::default();
    for n in &pool.order {
        if let Some((seg, _)) = n.split_once('/') {
            pkg_segments.insert(seg.to_string());
        }
    }
    // LOCAL-OK gate (round-59 Design B): a field-obscuring rename may
    // only fire when every class referencing the candidate lives inside
    // the root file's family, so the rename perturbs exactly ONE rendered
    // file. Ungated (Design A) the rule netted reqable −1653 / lark −5221
    // but weibo +36916: renames there flipped ambiguous type-vs-package
    // resolutions across thousands of pre-existing conflict families and
    // javac's error recovery turned "missing supertype, body suppressed"
    // files into full Object cascades. Lazy (progressive-browse) pools
    // skip the gate AND the rule — the multi-second image scan must not
    // stall single-class queries; the full-decompile pipeline always
    // materializes everything before installing renames.
    // Round-59 Design B stands: ungated field-clash renames were
    // retried AFTER the package-obscuring rules landed (weibo gate,
    // DDC_FC_UNGATED A/B) and still exploded to the 500k error cap —
    // fifth falsification of the ungated rename route.
    let local_ok: jdc_core::FxHashSet<String> = if pool_majority_materialized(pool) {
        refscan::local_ok(&pool.dexes, &field_clash_cands(pool, map))
    } else {
        jdc_core::FxHashSet::default()
    };
    let mut taken_sibling_displays: jdc_core::FxHashSet<String> =
        map.values().cloned().collect();
    for name in names {
        if !name.contains('$') || map.contains_key(name) {
            continue;
        }
        let Some(parent) = find_outer_name(pool, name) else {
            continue;
        };
        // Only a REAL `parent$rest` name renders as an inline member;
        // anonymous/local/lambda tails keep their flat own-file names.
        let Some(rest) = name
            .strip_prefix(parent.as_str())
            .and_then(|t| t.strip_prefix('$'))
        else {
            continue;
        };
        if !clean_member_tail(rest) {
            continue;
        }
        // Display simple names of every enclosing level, bottom-up; `cur`
        // ends at the ROOT (the top-level class whose file renders this
        // nested type — the scope the renamed tail lives in).
        let mut chain: Vec<String> = Vec::new();
        let mut cur = parent.clone();
        loop {
            let disp = map.get(&cur).cloned().unwrap_or_else(|| cur.clone());
            let simple = disp.rsplit('/').next().unwrap_or(&disp);
            chain.push(simple.rsplit('$').next().unwrap_or(simple).to_string());
            match find_outer_name(pool, &cur) {
                Some(o) if o != cur => cur = o,
                _ => break,
            }
        }
        let root: &str = &cur;
        let disp_parent = map
            .get(&parent)
            .cloned()
            .unwrap_or_else(|| parent.clone());
        let tail = rest.rsplit('$').next().unwrap_or(rest);
        let orphan = rest.contains('$');
        // Field obscuring (JLS 6.4.2): a nested type whose simple name
        // equals an enclosing-class field is OBSCURED by it in a
        // `<Parent>.<tail>` qualified reference — `io/flutter/view/g$i`
        // (a `static enum i`) renders `g.i.t`, but g also declares an
        // instance field `i`, so javac resolves `g.i` to the field:
        // "无法从静态上下文中引用非静态 变量 i" (975 of 1140 reqable
        // static-context errors). The fix renames the NESTED TYPE (the
        // registry carries every render path: shorten, inner_simple —
        // whose bypass was the round-55 blowup, fixed in jdc-core —
        // classdec declarations), gated to LOCAL-OK candidates by the
        // refscan above. PoC: in-file rename of flutter g's `enum i`
        // took the class from 2452 errors to 1.
        let field_clash = local_ok.contains(name.as_str());
        // Package shadow (JLS 6.4.2 at nested level): a nested type
        // DISPLAYED with a simple name equal to the first segment of a
        // package its FAMILY references binds every `s.x` qualified ref
        // in the root file to the member type — the family's own outer
        // rename can EXPOSE this (e/b → e/b2 un-collided the nested `b`,
        // which then hijacked `b.b2` — 循环继承). The round-47 candidate
        // guard only kept RENAMED tails off package segments; pre-
        // existing shadows need the rename too, family-gated: fire only
        // when this family's own descriptors reference the segment.
        // Orphans INCLUDED: an `Outer$$x` orphan normalizes to the
        // nested display `x` (k=1 invisible rename), and a clean tail
        // renders BARE in the family file — same package shadow as a
        // true nested (weixin li/u0$$r → `class r` hijacked `r.a` for
        // root package r).
        let pkg_shadow = fam_segs
            .get(root)
            .is_some_and(|segs| segs.contains(tail));
        let clash = chain.iter().any(|c| c == tail) || field_clash || pkg_shadow;
        if !orphan && !clash {
            continue;
        }
        // The enclosing class's field names (sanitized like the renderer)
        // — only needed once a rename actually fires, to keep the new
        // tail from being obscured again. Computed lazily so progressive
        // pools don't materialize parents for rules that don't rename.
        let mut enc_fields: jdc_core::FxHashSet<String> = jdc_core::FxHashSet::default();
        if field_clash || pkg_shadow {
            if let Some(pc) = pool.get(&parent) {
                for f in pc.static_fields.iter().chain(pc.instance_fields.iter()) {
                    enc_fields.insert(crate::classdec::java_ident(&f.name).into_owned());
                }
            }
        }
        let mut k = 0u32;
        loop {
            k += 1;
            let cand_tail = if k == 1 {
                tail.to_string()
            } else {
                format!("{tail}{k}")
            };
            // The candidate must clear every ancestor display name (javac
            // checks the whole chain) AND every enclosing field name (else
            // the renamed type is just obscured again). Sibling nested
            // tails are covered by the pool.has_name check below.
            if chain.contains(&cand_tail) || enc_fields.contains(&cand_tail) {
                continue;
            }
            // Package shadow: the renamed tail becomes a member type of
            // the root file's class, outranking any package of the same
            // name for qualified refs in that file. Same-package top-level
            // classes likewise outrank-then-break in-file simple refs.
            // BOTH checks apply to field-clash renames only: for the
            // orphan rule the k=1 candidate is the natural display name
            // (`g0$$r` → `g0$r` renders identically — an invisible rename),
            // and rejecting it forced weixin's whole `g0$$X` Runnable
            // family to visible `X17` renames, breaking every lucky
            // `g0.r`-style resolution corpus-wide (+2911 `变量 r` — the
            // orphan rule is ungated by design, predating LOCAL-OK).
            if field_clash || pkg_shadow {
                if pkg_segments.contains(&cand_tail) {
                    continue;
                }
                if pkg_shadow
                    && fam_segs
                        .get(root)
                        .is_some_and(|segs| segs.contains(cand_tail.as_str()))
                {
                    continue;
                }
                if let Some((root_pkg, _)) = root.rsplit_once('/') {
                    let sib = format!("{root_pkg}/{cand_tail}");
                    if pool.has_name(&sib) || taken_sibling_displays.contains(&sib)
                    {
                        // A top-level sibling exists — or an EARLIER rule
                        // minted that display (obscuring_class_renames
                        // moved ssosdk/b → ssosdk/b2 while this loop was
                        // about to mint nested $b → b2: two `b2`s in one
                        // scope, member type outranks the package sibling
                        // and hijacked every bare `b2.b(..)` call).
                        continue;
                    }
                    taken_sibling_displays.insert(sib);
                }
            }
            let cand = format!("{disp_parent}${cand_tail}");
            if pool.has_name(&cand) || assigned.contains(&cand) {
                continue;
            }
            assigned.insert(cand.clone());
            map.insert(name.clone(), cand);
            break;
        }
    }
}

/// Field-obscuring rename candidates: nested member types whose simple
/// name equals a field of the enclosing class, mapped to the top-level
/// class whose file renders them (the LOCAL-OK family root). Mirrors the
/// skip conditions of `nested_collision_renames` (same-name entries must
/// agree, or the gate proves locality for a rename that never fires — or
/// worse, skips one that does).
fn field_clash_cands(pool: &DexPool, map: &HashMap<String, String>) -> HashMap<String, String> {
    let mut out: HashMap<String, String> = HashMap::default();
    for name in &pool.order {
        if !name.contains('$') || map.contains_key(name) {
            continue;
        }
        let Some(parent) = find_outer_name(pool, name) else {
            continue;
        };
        let Some(rest) = name
            .strip_prefix(parent.as_str())
            .and_then(|t| t.strip_prefix('$'))
        else {
            continue;
        };
        if !clean_member_tail(rest) {
            continue;
        }
        let tail = rest.rsplit('$').next().unwrap_or(rest);
        let Some(pc) = pool.get(&parent) else {
            continue;
        };
        let clash = pc
            .static_fields
            .iter()
            .chain(pc.instance_fields.iter())
            .any(|f| crate::classdec::java_ident(&f.name).as_ref() == tail);
        if !clash {
            continue;
        }
        let mut root = parent.clone();
        loop {
            match find_outer_name(pool, &root) {
                Some(o) if o != root => root = o,
                _ => break,
            }
        }
        out.insert(name.clone(), root);
    }
    out
}

/// Whether the pool is (majority-)materialized: full-decompile pipelines
/// call `materialize_all` before installing renames, progressive browse
/// pools materialize a handful of classes on demand. The refscan gate
/// must not run (or stall startup) in the progressive case.
fn pool_majority_materialized(pool: &DexPool) -> bool {
    let (mut mat, mut lazy) = (0usize, 0usize);
    for e in pool.classes.values() {
        match e {
            ClassEntry::Eager(_) => mat += 1,
            ClassEntry::Lazy { pc, .. } => {
                if pc.get().is_some() {
                    mat += 1;
                } else {
                    lazy += 1;
                }
            }
        }
    }
    lazy == 0 || mat >= lazy
}

/// Compute and install the registry (call before worker threads spawn).
pub fn install_case_renames(pool: &DexPool) {
    let mut map = case_rename_map(pool);
    // Cross-package reference first segments (descriptor level). Lazy
    // (progressive-browse) pools skip the scan — materializing every
    // class must not stall single-class queries; the obscuring rules
    // then simply never fire (browse mode has no whole-program javac).
    let (pkg_segs, fam_segs) = if pool_majority_materialized(pool) {
        ref_segments(pool)
    } else {
        (
            jdc_core::FxHashMap::default(),
            jdc_core::FxHashMap::default(),
        )
    };
    // Package-leaf shadows FIRST: the nested-collision renames compute
    // display chains off the (renamed) parent display names — running
    // them before the parent rename left both rules minting the same
    // display (`a2` twice in weibo's AIDL families).
    pkg_leaf_shadow_renames(pool, &mut map);
    class_pkg_collision_renames(pool, &mut map);
    obscuring_class_renames(pool, &mut map, &pkg_segs);
    nested_collision_renames(pool, &mut map, &fam_segs);
    // LAST: lossy-sanitize collisions key on the display form every rule
    // above has already settled (`l.᩻ܶ` → `l/__.java`, 22,636 classes onto
    // 3 paths on bin.mt.plus — the silent-overwrite data loss).
    lossy_sanitize_renames(pool, &mut map);
    jdc_core::rename::set_class_renames(map);
    jdc_core::rename::set_field_renames(member_collision_renames(pool));
}

/// A top-level class `P/s` whose simple name equals the FIRST PACKAGE
/// SEGMENT of types that other classes in P reference cross-package.
/// Such a reference renders as the FQN `s.rest...`, and javac binds the
/// first segment to the nearest type in scope — the same-package class
/// `P/s` (JLS 6.4.2): `n91.f.a(x)` inside pc5/bd looks up member `f` OF
/// CLASS pc5.n91 (weixin's n0/n91/uc6/vc6 protobuf cluster: ~12k root
/// files, ~90k cascade lines; weibo's `a.g.b.a.b` chains root the same
/// way — class a beside subpackage a/ inside package a/g/b). No Java
/// syntax escapes an obscured first segment (imports can bypass it, but
/// single-type imports are blocked exactly where obfuscated leaf names
/// collide with same-package siblings), so the class moves: the r66
/// rename mechanism at a new trigger. Gated on DESCRIPTOR-level refs of
/// the candidate's own package (supers/fields/protos) — body-only refs
/// are missed by design; the import layer covers unblocked leaves there.
/// Descriptor-level cross-package reference first segments, aggregated
/// per PACKAGE (top-level obscurers: every file in P shares the scope)
/// and per FAMILY root (nested obscurers: a nested simple name is in
/// scope only inside its own root file). Raw `L..;` scans — no JavaType
/// parsing. Body-level refs are missed by design (the import layer
/// catches unblocked leaves there); descriptors cover supers, field
/// types and method protos — the header positions whose breakage starts
/// the error-type cascades.
/// Package (or family) → first segments of the cross-package types its
/// classes reference at descriptor level. Obscuring-rename trigger sets.
type SegMap<'a> = jdc_core::FxHashMap<&'a str, jdc_core::FxHashSet<&'a str>>;

fn ref_segments(pool: &DexPool) -> (SegMap<'_>, SegMap<'_>) {
    fn pkg_of(n: &str) -> &str {
        match n.rsplit_once('/') {
            Some((p, _)) => p,
            None => "",
        }
    }
    fn note<'a>(
        pkg_segs: &mut jdc_core::FxHashMap<&'a str, jdc_core::FxHashSet<&'a str>>,
        fam_segs: &mut jdc_core::FxHashMap<&'a str, jdc_core::FxHashSet<&'a str>>,
        pkg: &'a str,
        fam: &'a str,
        t: &'a str,
    ) {
        if !t.contains('/') {
            return; // root-package target: unreferenceable from named pkgs
        }
        let _ = pkg; // same-package refs COUNT: ddc renders headers/field
                     // types fully qualified (print_class_name), so `a.g.b.a.b`
                     // inside package a/g/b is obscured by class a/g/b/a just
                     // like a cross-package FQN (weibo's whole `a`-tree family).
        let seg = t.split('/').next().unwrap_or("");
        if !seg.is_empty() {
            pkg_segs.entry(pkg).or_default().insert(seg);
            fam_segs.entry(fam).or_default().insert(seg);
        }
    }
    fn scan<'a>(
        pkg_segs: &mut jdc_core::FxHashMap<&'a str, jdc_core::FxHashSet<&'a str>>,
        fam_segs: &mut jdc_core::FxHashMap<&'a str, jdc_core::FxHashSet<&'a str>>,
        pkg: &'a str,
        fam: &'a str,
        desc: &'a str,
    ) {
        let b = desc.as_bytes();
        let mut i = 0usize;
        while i < b.len() {
            if b[i] == b'L' {
                if let Some(end) = desc[i + 1..].find(';') {
                    note(pkg_segs, fam_segs, pkg, fam, &desc[i + 1..i + 1 + end]);
                    i += end + 2;
                    continue;
                }
            }
            i += 1;
        }
    }
    let mut pkg_segs: jdc_core::FxHashMap<&str, jdc_core::FxHashSet<&str>> =
        jdc_core::FxHashMap::default();
    let mut fam_segs: jdc_core::FxHashMap<&str, jdc_core::FxHashSet<&str>> =
        jdc_core::FxHashMap::default();
    for name in &pool.order {
        let Some(c) = pool.get(name) else { continue };
        let pkg = pkg_of(name);
        let fam = match name.find('$') {
            Some(i) => &name[..i],
            None => name.as_str(),
        };
        if let Some(sup) = &c.super_name {
            note(&mut pkg_segs, &mut fam_segs, pkg, fam, sup);
        }
        for sup in c.interfaces.iter() {
            note(&mut pkg_segs, &mut fam_segs, pkg, fam, sup);
        }
        for f in c.static_fields.iter().chain(c.instance_fields.iter()) {
            scan(&mut pkg_segs, &mut fam_segs, pkg, fam, &f.desc);
        }
        for m in c.direct_methods.iter().chain(c.virtual_methods.iter()) {
            scan(&mut pkg_segs, &mut fam_segs, pkg, fam, &m.desc);
        }
    }
    (pkg_segs, fam_segs)
}

/// A top-level class `P/s` whose simple name equals the FIRST PACKAGE
/// SEGMENT of types that classes in P reference cross-package. Such a
/// reference renders as the FQN `s.rest...`, and javac binds the first
/// segment to the nearest type in scope — the same-package class `P/s`
/// (JLS 6.4.2, no package fallback — verified): `n91.f.a(x)` inside
/// pc5/bd looks up member `f` OF CLASS pc5.n91 (weixin's n0/n91/uc6/vc6
/// protobuf cluster: ~12k root files, ~90k cascade lines; weibo's
/// `a.g.b.a.b` chains root the same way). No Java syntax escapes an
/// obscured first segment, so the class moves: the r66 rename mechanism
/// at a new trigger, gated on DESCRIPTOR-level refs of the candidate's
/// package (body-only refs go to the import layer). Nested obscurers
/// (a nested simple name shadowing a package inside its family file)
/// are the companion trigger in nested_collision_renames.
fn obscuring_class_renames(
    pool: &DexPool,
    map: &mut HashMap<String, String>,
    pkg_segs: &SegMap,
) {
    fn pkg_of(n: &str) -> &str {
        match n.rsplit_once('/') {
            Some((p, _)) => p,
            None => "",
        }
    }
    // Root package first segments (segments some class lives under).
    let mut root_segs: jdc_core::FxHashSet<&str> = jdc_core::FxHashSet::default();
    for n in &pool.order {
        if let Some(i) = n.find('/') {
            root_segs.insert(&n[..i]);
        }
    }
    let mut cands: Vec<&String> = Vec::new();
    for name in &pool.order {
        let simple = name.rsplit('/').next().unwrap_or(name);
        let pkg = pkg_of(name);
        if simple.is_empty() || simple.contains('$') || pkg.is_empty() {
            continue;
        }
        if !root_segs.contains(simple) {
            continue;
        }
        if map.get(name).is_some_and(|v| v != name) {
            continue; // an earlier rule moved it
        }
        // NB: NO "class lives under the s tree" exclusion. weibo's
        // a/a/b/c/l/a shadowed the FQN refs `a.a.a.a2` that its own
        // package files render for cross-subpackage targets (the
        // starts_with("a/") skip left the WHOLE a-tree unrenamed —
        // 找不到符号 类 a chains). The two-stage refsegs gates below
        // require an actual cross-package FQN ref for the trigger.
        cands.push(name);
    }
    let ncands = cands.len();
    if std::env::var("DDC_STATS").is_ok() {
        eprintln!(
            "[renames] obscuring candidates={ncands} pkgs-with-segs={}",
            pkg_segs.len()
        );
    }
    if cands.is_empty() {
        return;
    }
    // Fast collision oracle: sorted names + prefix range scan (the r66
    // candidate loop's `order.iter().any(starts_with)` is O(n) per
    // candidate — this rule can mint tens of thousands of renames).
    let mut sorted: Vec<&String> = pool.order.iter().collect();
    sorted.sort_unstable();
    let prefix_free = |cand: &str| -> bool {
        // No class named cand, nothing nested under cand$..., no package cand/...
        let idx = sorted.partition_point(|n| n.as_str() < cand);
        !sorted[idx..].first().is_some_and(|n| n.starts_with(cand))
    };
    let pkg_simples = pool.package_simples();
    let mut taken_displays: jdc_core::FxHashSet<String> = map.values().cloned().collect();
    let mut renamed = 0usize;
    // Stage 1: descriptor-level hits rename now; the rest queue for the
    // BODY-level image scan (weixin pc5's `invoke-static Ln91/f;.a` —
    // 7.8k+6.6k error lines the descriptor gate cannot see).
    let mut stage2: Vec<&String> = Vec::new();
    for name in cands {
        let pkg = pkg_of(name);
        let simple = name.rsplit('/').next().unwrap_or(name);
        if !pkg_segs.get(pkg).is_some_and(|segs| segs.contains(simple)) {
            stage2.push(name);
            continue;
        }
        let siblings = pkg_simples.get(pkg);
        let mut k = 1u32;
        loop {
            k += 1;
            let new_simple = format!("{simple}{k}");
            let cand = if pkg.is_empty() {
                new_simple.clone()
            } else {
                format!("{pkg}/{new_simple}")
            };
            // Exact/prefix pool clash, another rule's display target, the
            // package's own referenced segments (the new name must not
            // obscure ANOTHER package the same files reference), or a
            // case-insensitive clash with a same-package sibling
            // (case-insensitive filesystems fold the files).
            let clash = !prefix_free(&cand)
                || taken_displays.contains(&cand)
                || pkg_segs
                    .get(pkg)
                    .is_some_and(|segs| segs.contains(new_simple.as_str()))
                || siblings.is_some_and(|s| {
                    s.iter()
                        .any(|p| p.eq_ignore_ascii_case(&new_simple) && *p != simple)
                });
            if !clash {
                taken_displays.insert(cand.clone());
                map.insert(name.clone(), cand);
                renamed += 1;
                break;
            }
        }
    }
    let mut body_renamed = 0usize;
    if !stage2.is_empty() && pool_majority_materialized(pool) {
        let stage2_pkgs: jdc_core::FxHashSet<String> =
            stage2.iter().map(|n| pkg_of(n).to_string()).collect();
        let body_segs = crate::refscan::body_ref_segments(&pool.dexes, &stage2_pkgs);
        for name in stage2 {
            let pkg = pkg_of(name);
            let simple = name.rsplit('/').next().unwrap_or(name);
            if !body_segs
                .get(pkg)
                .is_some_and(|segs| segs.contains(simple))
            {
                continue;
            }
            let siblings = pkg_simples.get(pkg);
            let mut k = 1u32;
            loop {
                k += 1;
                let new_simple = format!("{simple}{k}");
                let cand = format!("{pkg}/{new_simple}");
                let clash = !prefix_free(&cand)
                    || taken_displays.contains(&cand)
                    || pkg_segs
                        .get(pkg)
                        .is_some_and(|segs| segs.contains(new_simple.as_str()))
                    || body_segs
                        .get(pkg)
                        .is_some_and(|segs| segs.contains(new_simple.as_str()))
                    || siblings.is_some_and(|s| {
                        s.iter()
                            .any(|p| p.eq_ignore_ascii_case(&new_simple) && *p != simple)
                    });
                if !clash {
                    taken_displays.insert(cand.clone());
                    map.insert(name.clone(), cand);
                    renamed += 1;
                    body_renamed += 1;
                    break;
                }
            }
        }
    }
    if std::env::var("DDC_STATS").is_ok() {
        eprintln!(
            "[renames] obscuring-class renames: {renamed} (body-gated {body_renamed}; pkgs-with-segs={}, cands={ncands})",
            pkg_segs.len()
        );
    }
}

/// Class-level counterpart of the non-ASCII rule already applied to
/// MEMBERS by `member_collision_renames`, kept as the SAFETY NET behind
/// the injective `sanitize_fq`.
///
/// `sanitize_fq` escapes non-identifier characters as `_u<hex>`, so two
/// distinct classes normally reach two distinct file names. It cannot be
/// injective against *lookalikes*, because ident-safe ASCII must pass
/// through unchanged: a literal class named `_u1a7b` collides with the
/// class whose single character is U+1A7B, and the pre-existing keyword
/// repair makes `_do` collide with `do`. Without this pass those pairs
/// repeat the original disaster — the writer opens with `create_new`,
/// takes EEXIST, unlinks and rewrites, so one class is destroyed with
/// exit code 0 (on bin.mt.plus the lossy predecessor of this rule hid
/// 22,633 of 30,768 classes exactly this way).
///
/// Groups are keyed by the SANITIZED path — the exact string both the
/// declaration site (`sanitize_ref`) and the writer (`source_path`)
/// produce. The first (sorted) member of each group keeps its name; the
/// rest become `<sanitized>_2`, `<sanitized>_3`, … Pure ASCII, so a
/// minted name is a fixed point of the sanitizer: it cannot silently
/// join a second collision group, and the declaration, every reference
/// and the file name agree by construction through `apply_class_rename`.
///
/// Costs nothing on a healthy pool: one pass over the emission set, and
/// it returns before touching the registry when every key is unique.
/// Runs LAST in `install_case_renames`: it keys on the display form the
/// earlier rules have already settled.
fn lossy_sanitize_renames(pool: &DexPool, map: &mut HashMap<String, String>) {
    let names = top_level_classes(pool);
    // Current display of an emission-set name. `case_rename_map` has
    // already inserted an entry for every one of these (identity or
    // renamed) and an exact registry hit short-circuits
    // `apply_class_rename`, so `map` already holds the final display —
    // no `$`-prefix walk is needed here.
    fn key_of(map: &HashMap<String, String>, n: &str) -> String {
        let disp = map.get(n).map(|s| s.as_str()).unwrap_or(n);
        sanitize_internal(disp)
    }
    let mut counts: HashMap<String, u32> = HashMap::default();
    for n in &names {
        *counts.entry(key_of(map, n)).or_insert(0) += 1;
    }
    if counts.values().all(|c| *c == 1) {
        return;
    }
    // Every sanitized path in the POOL is off-limits for a minted name,
    // not just the emission set: an inlined nested class still owns the
    // display it renders under inside its outer's file.
    let mut taken: jdc_core::FxHashSet<String> = pool
        .order
        .iter()
        .map(|n| sanitize_internal(map.get(n).map(|s| s.as_str()).unwrap_or(n)))
        .collect();
    let mut sorted: Vec<&String> = names.iter().collect();
    sorted.sort();
    let mut seen: jdc_core::FxHashSet<String> = jdc_core::FxHashSet::default();
    let mut renamed = 0usize;
    for n in sorted {
        let key = key_of(map, n);
        if counts.get(&key).copied().unwrap_or(0) < 2 {
            continue;
        }
        if seen.insert(key.clone()) {
            continue; // first (sorted) member of the group keeps its name
        }
        let (pkg, simple) = match key.rsplit_once('/') {
            Some((p, s)) => (p, s),
            None => ("", key.as_str()),
        };
        let mut i = 1u32;
        loop {
            i += 1;
            let cand = if pkg.is_empty() {
                format!("{simple}_{i}")
            } else {
                format!("{pkg}/{simple}_{i}")
            };
            if !taken.insert(cand.clone()) {
                continue; // already owned by a real class or another mint
            }
            // The minted display IS the sanitized name, so
            // sanitize_internal(cand) == cand and the writer's file name
            // matches the declaration it writes into it.
            map.insert(n.clone(), cand);
            renamed += 1;
            break;
        }
    }
    if std::env::var("DDC_STATS").is_ok() {
        eprintln!("[renames] lossy-sanitize collisions renamed={renamed}");
    }
}

/// A class named `P/s` while OTHER classes live under `P/s/` — the dex
/// namespace is flat, so obfuscators freely mint a class whose simple
/// name equals a subpackage of its own package (`ptr.a` +
/// `ptr.a.a`, weibo rsplay/ptr families). That namespace is
/// unrepresentable in Java: the declaration itself is
/// "类 a与带有相同名称的程序包冲突" and every qualified reference
/// through the segment becomes unresolvable (the 53.8k weibo
/// missing-class errors root here, with Object/getClass cascades on
/// top). Rename the class — the registry carries declarations and
/// references — with a candidate that cannot recreate the shape (not a
/// class, not a package, not a nested-class prefix, not another rule's
/// target).
fn class_pkg_collision_renames(pool: &DexPool, map: &mut HashMap<String, String>) {
    // Every package that exists in the pool — ALL ancestor prefixes of
    // every class path, not just direct parents: Java sees package
    // `z` as existing when any class lives at `z/a/b` (the weibo
    // `com.sina.weibo.z` interface collides with a subpackage-only
    // `z/` that has no directly-resident class).
    let mut pkgs: jdc_core::FxHashSet<String> = jdc_core::FxHashSet::default();
    for n in &pool.order {
        let mut rest = n.as_str();
        while let Some(i) = rest.rfind('/') {
            rest = &rest[..i];
            pkgs.insert(rest.to_string());
        }
    }
    for name in &pool.order {
        let Some((pkg, simple)) = name.rsplit_once('/') else {
            continue;
        };
        if simple.is_empty() || pkg.is_empty() || simple.contains('$') {
            // Nested classes render dotted (`Outer.a`) — no package
            // effect from their `$` tail.
            continue;
        }
        if !pkgs.contains(name) || map.get(name).is_some_and(|v| v != name) {
            continue; // no subpackage under it, or an earlier rule moved it
        }
        let mut k = 1u32;
        loop {
            k += 1;
            let cand = format!("{pkg}/{simple}{k}");
            let clash = pkgs.contains(&cand)
                || pool.order.iter().any(|n| n.starts_with(&cand))
                || map.values().any(|v| *v == cand);
            if !clash {
                map.insert(name.clone(), cand);
                break;
            }
        }
    }
}

/// Obfuscators can name a class after its own package leaf (`package k;`
/// `class k extends EditText`). Every SAME-PACKAGE reference to `k.Anything`
/// then resolves the first segment to the CLASS, not the package —
/// javac looks for `Anything` as a member of class k ("找不到符号 class p2,
/// location: class k"). Renaming the class (display-level, the registry
/// carries declarations and references) makes the package the only
/// thing the leaf name resolves to.
fn pkg_leaf_shadow_renames(pool: &DexPool, map: &mut HashMap<String, String>) {
    let taken: jdc_core::FxHashSet<String> = pool.order.iter().cloned().collect();
    for name in &pool.order {
        let Some((pkg, simple)) = name.rsplit_once('/') else {
            continue;
        };
        if simple.is_empty() || pkg.is_empty() {
            continue;
        }
        let leaf = pkg.rsplit('/').next().unwrap_or(pkg);
        // class name == package leaf, and the package really exists
        // (more classes than just this one live under it).
        if simple != leaf {
            continue;
        }
        let prefix = format!("{pkg}/");
        let siblings = pool
            .order
            .iter()
            .filter(|n| n.starts_with(&prefix) && **n != *name)
            .count();
        if siblings == 0 || map.get(name).is_some_and(|v| v != name) {
            continue; // already renamed by an earlier rule; identity anchors are fine to overwrite
        }
        let mut k = 1u32;
        loop {
            k += 1;
            let cand = format!("{pkg}/{simple}{k}");
            if !taken.contains(&cand) && !map.values().any(|v| *v == cand) {
                map.insert(name.clone(), cand);
                break;
            }
        }
    }
}

/// Member-level collision renames (fields AND methods of one class).
/// Two sources collapse onto one display name: obfuscators renaming a
/// synthetic outer reference (`this$0`) onto a real field's name, and
/// the identifier sanitizer mapping non-ASCII to `_` (whole
/// `ERROR_中文` families render one `ERROR________`).
/// Both are legal in bytecode (members resolve by index) and both are
/// javac "already defined" errors in source. Policy per colliding group
/// (fields keep the first non-synthetic name; a synthetic outer
/// reference restores `this$0`; everything else gets `2`/`3` suffixes;
/// method keys include erased parameter types):
/// Keys stay ORIGINAL (name, descriptor) so references resolve exactly;
/// only the rendered identifier changes.
fn member_collision_renames(
    pool: &DexPool,
) -> HashMap<std::sync::Arc<str>, Vec<jdc_core::rename::FieldRename>> {
    let mut out: HashMap<std::sync::Arc<str>, Vec<jdc_core::rename::FieldRename>> =
        HashMap::default();
    // Direct nested tails per base class (`C$a` → a for base C), for
    // the companion-field rule below.
    let mut child_tails: jdc_core::FxHashMap<&str, Vec<&str>> =
        jdc_core::FxHashMap::default();
    for n in &pool.order {
        if let Some(i) = n.find('$') {
            let base = &n[..i];
            let rest = &n[i + 1..];
            let tail = match rest.find('$') {
                Some(j) => &rest[..j],
                None => rest,
            };
            if !tail.is_empty() {
                child_tails.entry(base).or_default().push(tail);
            }
        }
    }
    for name in &pool.order {
        let Some(pc) = pool.get_if_materialized(name) else {
            continue;
        };
        let outer_ref_desc = find_outer_name(pool, name)
            .filter(|o| o != name)
            .map(|o| format!("L{o};"));
        // ---- fields: one shared namespace (static + instance) ----
        let fields: Vec<&PoolField> = pc
            .static_fields
            .iter()
            .chain(pc.instance_fields.iter())
            .collect();
        let fgroups: HashMap<String, Vec<&PoolField>> = HashMap::default();
        let mut fgroups = fgroups;
        for f in &fields {
            let disp = crate::classdec::java_ident(&f.name).into_owned();
            fgroups.entry(disp).or_default().push(f);
        }
        let f_taken: jdc_core::FxHashSet<String> = fields
            .iter()
            .map(|f| crate::classdec::java_ident(&f.name).into_owned())
            .collect();
        let mut f_taken = f_taken;
        for group in fgroups.values() {
            if group.len() < 2 {
                continue;
            }
            let keeper_pos = group
                .iter()
                .position(|f| f.access & crate::access::ACC_SYNTHETIC == 0)
                .unwrap_or(0);
            for (gi, f) in group.iter().enumerate() {
                if gi == keeper_pos {
                    continue;
                }
                let base = crate::classdec::java_ident(&f.name).into_owned();
                let display: String = match &outer_ref_desc {
                    Some(ord)
                        if f.access & crate::access::ACC_SYNTHETIC != 0
                            && !f.is_static
                            && *ord == f.desc =>
                    {
                        let mut k = 0u32;
                        loop {
                            let cand = if k == 0 {
                                "this$0".to_string()
                            } else {
                                format!("this$0{k}")
                            };
                            if f_taken.insert(cand.clone()) {
                                break cand;
                            }
                            k += 1;
                        }
                    }
                    _ => suffix_unique(&base, &mut f_taken),
                };
                out.entry(std::sync::Arc::from(name.as_str()))
                    .or_default()
                    .push(jdc_core::rename::FieldRename {
                        name: std::sync::Arc::from(f.name.as_str()),
                        desc: std::sync::Arc::from(f.desc.as_str()),
                        display: std::sync::Arc::from(display.as_str()),
                    });
            }
        }
        // ---- companion-holder fields vs nested types (JLS 6.4.2 in
        // expression context): Kotlin compiles `object`/companion
        // holders as a static field whose name EQUALS the nested class
        // it instances (`static final j$a a`), and static members of the
        // nested are invoked as `j.a.c(..)` — but `j.a` in an EXPRESSION
        // resolves to the FIELD (variables outrank member types), so the
        // call becomes a member lookup on the instance type... which is
        // the same class, yet javac reports 找不到符号 for statics
        // reached through the shadowed path, and the whole file
        // error-types (weibo feed/business/j: 654 errors + closure
        // cascade; the nested-side rename for this is the LOCAL-OK-gated
        // rule that explodes ungated — the FIELD side is safe because
        // dex field refs always carry the declaring class, so this
        // registry keys every consumer). Gate: the nested class has
        // static members (companion shape) and the field's descriptor
        // is exactly the nested type.
        if let Some(tails) = child_tails.get(name.as_str()) {
            for tail in tails {
                let nested = format!("{name}${tail}");
                let Some(nc) = pool.get_if_materialized(&nested) else {
                    continue;
                };
                // Only CLEAN tails render dotted (`j.a`); flat-rendered
                // tails (`j$1`) carry the `$` and never clash.
                if !clean_member_tail(tail) {
                    continue;
                }
                let has_statics = nc
                    .static_fields
                    .iter()
                    .chain(nc.instance_fields.iter())
                    .any(|f| f.is_static)
                    || nc.all_methods().any(|m| m.access & crate::access::ACC_STATIC != 0);
                if !has_statics {
                    continue;
                }
                // STRICT companion shape: the field's descriptor IS the
                // nested type (`static final j$a a`). The relaxed form
                // (any same-named field beside a statics-bearing nested)
                // minted thousands of renames on weibo and exploded the
                // battery to the 500k cap — the field funnel does not
                // hold at that scale (改名面 vs 引用覆盖面, again).
                let want_desc = format!("L{nested};");
                for f in pc.static_fields.iter().chain(pc.instance_fields.iter()) {
                    if f.desc != want_desc {
                        continue;
                    }
                    let disp = crate::classdec::java_ident(&f.name).into_owned();
                    if disp != crate::classdec::java_ident(tail) {
                        continue;
                    }
                    // Already renamed by the duplicate-group rule above?
                    let already = out
                        .get(&std::sync::Arc::from(name.as_str()))
                        .is_some_and(|v| {
                            v.iter().any(|fr| *fr.name == *f.name && *fr.desc == *f.desc)
                        });
                    if already {
                        continue;
                    }
                    let display = suffix_unique(&disp, &mut f_taken);
                    f_taken.insert(display.clone());
                    out.entry(std::sync::Arc::from(name.as_str()))
                        .or_default()
                        .push(jdc_core::rename::FieldRename {
                            name: std::sync::Arc::from(f.name.as_str()),
                            desc: std::sync::Arc::from(f.desc.as_str()),
                            display: std::sync::Arc::from(display.as_str()),
                        });
                }
            }
        }
        // ---- methods: display key = sanitized name + erased params ----
        let methods: Vec<&PoolMethod> = pc.all_methods().collect();
        let mut m_taken: jdc_core::FxHashSet<String> = methods
            .iter()
            .map(|m| crate::classdec::java_ident(&m.name).into_owned())
            .collect();
        let mut mgroups: HashMap<(String, String), Vec<&PoolMethod>> = HashMap::default();
        for m in &methods {
            let d: &str = &m.desc;
            let lo = d.find('(').map(|i| i + 1).unwrap_or(0);
            let hi = d.find(')').unwrap_or(d.len());
            let key = (
                crate::classdec::java_ident(&m.name).into_owned(),
                d[lo..hi].to_string(),
            );
            mgroups.entry(key).or_default().push(m);
        }
        for ((base, _), group) in &mgroups {
            if group.len() < 2 {
                continue;
            }
            // A COVARIANT override pair (same ORIGINAL name, different
            // descriptors — the compiler's bridge shape: the interface's
            // `deserialize(e)` plus the narrower implementation). Renaming
            // either side breaks @Override ("is not abstract and does not
            // override abstract method deserialize(e)", ~2.2k on reqable);
            // the emitter's claim logic already renders only the
            // non-bridge member of the pair.
            let covariant = group
                .iter()
                .any(|m| group.iter().any(|o| o.name == m.name && o.desc != m.desc));
            if covariant {
                continue;
            }
            // Real re-declarations (same original name AND descriptor —
            // R8 duplicates) are already deduped by the emitter's claim
            // logic; only SANITIZER collapses reach here and every one of
            // them is a distinct method.
            let mut base = base.clone();
            let mut seen_orig: jdc_core::FxHashSet<(&str, &str)> =
                jdc_core::FxHashSet::default();
            for m in group {
                if !seen_orig.insert((&m.name, &m.desc)) {
                    continue; // exact duplicate: emitter drops it
                }
                let display = suffix_unique(&base, &mut m_taken);
                out.entry(std::sync::Arc::from(name.as_str()))
                    .or_default()
                    .push(jdc_core::rename::FieldRename {
                        name: m.name.clone(),
                        desc: m.desc.clone(),
                        display: std::sync::Arc::from(display.as_str()),
                    });
                // The base stays the sanitized ORIGINAL so the first
                // occurrence keeps the plain name.
                let _ = &mut base;
            }
        }
    }
    out
}

fn suffix_unique(base: &str, taken: &mut jdc_core::FxHashSet<String>) -> String {
    let mut k = 1u32;
    loop {
        k += 1;
        let cand = format!("{base}{k}");
        if taken.insert(cand.clone()) {
            return cand;
        }
    }
}

/// Install member renames (call with the class rename install, before
/// workers spawn).
pub fn install_field_renames(pool: &DexPool) {
    jdc_core::rename::set_field_renames(member_collision_renames(pool));
}
pub use jdc_core::rename::apply_class_rename;
