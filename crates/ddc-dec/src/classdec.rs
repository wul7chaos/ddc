//! Class-level rendering: header, fields (with static initializers),
//! constructors, methods and nested member classes.

use jdc_core::emit::{escape_string, format_float, Printer};
use jdc_core::types::JavaType;
use jdc_core::Ctx;

use crate::access::*;
use crate::PoolField;
use crate::ctx::{java_type_to_generic, DexCtx};
use crate::method::decompile_method;
use crate::{desc_type, DexPool, PoolClass, PoolMethod, StaticValue};
use ddc_dex::insn::InsnKind;

#[derive(Debug, Clone)]
pub struct ClassOptions {
    /// Prefix each file with a provenance comment.
    pub provenance: bool,
}

impl Default for ClassOptions {
    fn default() -> Self {
        ClassOptions { provenance: true }
    }
}

/// Decompile one class (plus nested member classes) to a Java source file.
///
/// Classes with large method bodies run in a monitored thread with a
/// deadline: pathological CFGs can drive the shared structurer's walk into
/// an exponential exploration that never returns. On timeout the class is
/// reported failed and the thread is abandoned (reaped at process exit).
/// One registered monitored decompile: the receiver the worker polls at
/// the tail, the class name (for diagnostics), and the deadline counted
/// from the spawn.
pub type PendingMonitor = (
    std::sync::mpsc::Receiver<Result<String, String>>,
    String,
    std::time::Instant,
);

pub fn decompile_class(
    pool: &std::sync::Arc<DexPool>,
    class: &PoolClass,
    opts: &ClassOptions,
    pending: &std::sync::Mutex<Vec<PendingMonitor>>,
) -> anyhow::Result<String> {
    if class_is_risky(pool, class) {
        // Detached monitored thread: the CALLER registers the receiver and
        // moves on (awaiting happens at the end of the run) — a spinning
        // pathological method no longer stalls its worker.
        let pool2 = pool.clone();
        let cls = class.clone();
        let opts2 = opts.clone();
        let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();
        let name = class.name.clone();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let _ = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let r = decompile_class_impl(&pool2, &cls, &opts2).map_err(|e| format!("{:#}", e));
                let _ = tx.send(r);
            });
        pending.lock().unwrap().push((rx, name, deadline));
        return Err(anyhow::anyhow!(
            "deferred to monitored thread (pathological-CFG guard)"
        ));
    }
    decompile_class_impl(pool, class, opts)
}

/// Any method in the exponential-walk danger zone → run under the
/// deadline. Calibrated between real workloads: routine R8 classes top out
/// near 11k insns (reqable's biggest `<clinit>`); the exponential hangs
/// (weibo's gson TypeAdapters) sit at 24k+.
fn class_is_risky(pool: &DexPool, class: &PoolClass) -> bool {
    // Header peek only — decoding every body here would double the APK's
    // total decode work.
    for m in class.all_methods() {
        if m.code_off == 0 {
            continue;
        }
        if let Some(dex) = pool.dex(m.dex_idx) {
            if let Some((_regs, insns)) = dex.code_stats(m.code_off) {
                if insns > 16_000 {
                    return true;
                }
            }
        }
    }
    false
}

fn decompile_class_impl(
    pool: &DexPool,
    class: &PoolClass,
    opts: &ClassOptions,
) -> anyhow::Result<String> {
    let ctx = DexCtx::new(pool, class);
    // Size the class buffer up front: corpus-average classes render to
    // ~1KB per method — without the reserve the String doubles through
    // 4-6 realloc+copy rounds per class (~2× the final size in memmove).
    // Cap the reserve: weixin's monster classes (hundreds of methods)
    // would reserve megabytes of untouched capacity per class — large
    // blocks take mimalloc's commit/purge path (madvise churn showed up
    // in profiles). 256KB covers the corpus-average class dozens of
    // times over; bigger classes pay a few extra doublings.
    let mut out = String::with_capacity(
        (class.all_methods().count() * 1024).min(256 * 1024) + 512,
    );
    if opts.provenance {
        static BANNER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        out.push_str(BANNER.get_or_init(|| {
            format!(
                "// Decompiled by https://github.com/ejfkdev/ddc {}\n",
                env!("CARGO_PKG_VERSION")
            )
        }));
        // Provenance: which input image this class was lifted from (jadx's
        // `loaded from: classes.dex` pattern). Byte-stable across runs —
        // deliberately NO timestamp so outputs diff cleanly. Direct
        // push_str: the per-class format! temporaries were 3 allocations
        // × every class in the corpus.
        if let Some(dex) = pool.dex(class.dex_idx) {
            if let Some(label) = pool.dex_labels.get(class.dex_idx) {
                out.push_str("// From: ");
                out.push_str(label);
                out.push_str(" (DEX ");
                out.push_str(&dex.version);
                out.push_str(")\n");
            }
        }
    }
    if let Some(src) = &class.source_file {
        out.push_str("// Source file: ");
        out.push_str(src);
        out.push('\n');
    }
    if class.is_synthetic() {
        out.push_str("// synthetic\n");
    }
    let (pkg, _) = split_name(&class.name);
    // Two-pass import assembly: the body renders FIRST (into a temp
    // buffer) with the obscured render state installed — expression
    // positions DISCOVER additional obscured refs on the fly — then the
    // file assembles as provenance + package + imports + body. The one
    // extra body copy is the price of correct import placement.
    let mut obscured_map = compute_obscured_renders(pool, class);
    // Same-package simple-name collisions: an import shadows every
    // same-package use of that simple in this file — drop those from
    // the map (their refs stay qualified, erroring honestly).
    let blocked: jdc_core::FxHashSet<String> = pool
        .package_simples()
        .get(&pkg)
        .cloned()
        .unwrap_or_default();
    obscured_map.retain(|_internal, simple| !blocked.contains(simple));
    set_obscured_state(class.name.clone(), obscured_map.clone(), blocked.clone());
    let mut body_buf = String::with_capacity(out.capacity() / 2);
    let body_res = emit_class_body(pool, class, &ctx, opts, &mut body_buf, 0);
    let recorded = take_recorded_and_clear();
    body_res?;
    if !pkg.is_empty() {
        out.push('\n');
        out.push_str("package ");
        let dotted = pkg.replace('/', ".");
        out.push_str(&sanitize_fq(&dotted));
        out.push_str(";\n");
        // Imports: the metadata map plus the expression-level
        // discoveries, pool-known only (an import of an unknown class
        // is a hard error), display-renamed, sorted for determinism.
        let mut import_set: jdc_core::FxHashSet<String> =
            obscured_map.keys().cloned().collect();
        for r in recorded {
            if pool.get(&r).is_some() {
                let simple = r.rsplit('/').next().unwrap_or("");
                if !blocked.contains(simple) {
                    import_set.insert(r);
                }
            }
        }
        let mut imports: Vec<String> = import_set
            .iter()
            .map(|internal| crate::classdec::dotted(&crate::apply_class_rename(internal)))
            .collect();
        imports.sort();
        for display in imports {
            out.push_str(&format!("import {};\n", display));
        }
    }
    out.push('\n');
    out.push_str(&body_buf);
    Ok(out)
}

/// Render the class header, fields, methods and nested member classes into
/// `out` (used at top level and recursively for nested members).
// `opts` is only consumed by the nested-member recursion below — that
// is its purpose (propagating emission options into inline children).
/// One enum constant: the source identifier plus constructor
/// arguments beyond the compiler-mandated `(String name, int ordinal)`.
use jdc_core::ir::expr::{ConstVal, Expr};
use jdc_core::ir::stmt::Stmt;
struct EnumConst {
    #[allow(dead_code)]
    field: String,
    name: String,
    extra_args: Vec<Expr>,
}

/// Collect enum constants for a true `enum` rendering. Every ACC_ENUM
/// static field must be initialized in `<clinit>` by
/// `Self.field = new Self("NAME", ordinal, ...)` — the shape javac/d8
/// always emit. Returns None (caller falls back to the desugared
/// `/* enum */ class` form) when anything is missing: R8 variance,
/// constant-specific bodies (the field holds an anonymous subclass), or
/// a <clinit> that failed to decompile.
fn collect_enum_constants(
    pool: &DexPool,
    class: &PoolClass,
    ctx: &DexCtx<'_>,
) -> Option<(Vec<EnumConst>, crate::method::MethodBody)> {
    let _ = ctx;
    let const_fields: Vec<&PoolField> = class
        .static_fields
        .iter()
        .filter(|f| f.access & crate::access::ACC_ENUM != 0)
        .collect();
    if const_fields.is_empty() { return None; }
    let clinit = class.all_methods().find(|m| &*m.name == "<clinit>")?;
    let mut body = decompile_method(pool, class, clinit).ok().flatten()?;

    // R8/d8 split the constant build across an intermediate local:
    //   Self v0 = new Self("NAME", i, ...);
    //   a = v0;                       // sput to the ACC_ENUM field
    //   Self[] v5 = new Self[n]; v5[k] = v0; ...; $VALUES = v5;
    // Pass 1 registers each intermediate (or direct) new; pass 2 binds
    // them to the ACC_ENUM fields; references to the locals are then
    // rewritten into constant identifiers so the $VALUES build keeps
    // compiling once the definitions drop.
    use std::collections::HashMap;
    let mut const_name: Vec<String> = Vec::new();
    let mut const_field: Vec<String> = Vec::new();
    let mut const_extra: Vec<Vec<Expr>> = Vec::new();
    // Declared ordinal per constant (the ctor's int arg) — the merge at
    // the end orders field-bound and synthetic constants by it and
    // demands a dense 0..n run (Java derives ordinals from declaration
    // position).
    let mut const_ord: Vec<i64> = Vec::new();
    let mut var_of: HashMap<u32, usize> = HashMap::default(); // local id -> const idx
    let mut drop_stmts: Vec<usize> = Vec::new();

    // Collect (immutable borrows) first; the mutable passes come after.
    if !matches!(&body.body, Stmt::Block(_)) { return None; }

    // Pass 1: definitions. (immutable borrow; rewrite comes later)
    // Rolling reaching-defs of clinit locals for resolving enum-ctor
    // extra args (resolve_enum_arg); `tainted` flips at the first
    // non-straight-line statement — past it, linear-scan defs are no
    // longer provable and Local args reject the enum mode.
    let mut defs: HashMap<u32, &Expr> = HashMap::default();
    let mut tainted = false;
    let self_name: std::sync::Arc<str> = class.name.as_str().into();
    let self_ty =
        jdc_core::ir::expr::TypeRef::J(JavaType::Object(class.name.as_str().into()));
    // A REUSED intermediate local: `v = new Self(..); A = v; v = new
    // Self(..); B = v;` — the first def is a LocalDef, the rest are
    // Assigns to the same var. Register every def site; pass 2 resolves
    // a local sput to the def site CURRENT at that statement (rolling),
    // not to the var (Kotlin EnumEntries enums, RegexOption: 8→7 count
    // mismatch used to abort the whole promotion).
    let mut def_site: HashMap<usize, (u32, usize)> = HashMap::default();
    for (i, st) in match &body.body {
        Stmt::Block(v) => v.iter().enumerate(),
        _ => return None,
    } {
        // (var, New) for both def shapes: LocalDef and Assign-to-local.
        let def_shape: Option<(u32, &Expr)> = match st {
            Stmt::LocalDef {
                var,
                init: Some(e),
                ..
            } => Some((*var, e)),
            Stmt::ExprStmt(Expr::Assign { target, value, .. }) => {
                match (&**target, &**value) {
                    (Expr::Local { var, .. }, e) => Some((*var, e)),
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some((var, Expr::New { cls: ncls, args, .. })) = def_shape {
            if ncls.as_ref() == class.name && args.len() >= 2 {
                if let (Expr::Const(ConstVal::Str(n)), Expr::Const(ConstVal::Int(ord0))) =
                    (&args[0], &args[1])
                {
                    if java_ident(n).as_ref() != &**n || n.is_empty() { return None; }
                    var_of.insert(var, const_name.len());
                    def_site.insert(i, (var, const_name.len()));
                    const_ord.push(*ord0 as i64);
                    const_name.push(n.to_string());
                    const_field.push(String::new());
                    const_extra.push(resolve_enum_extras(
                        &args[2..],
                        &defs,
                        tainted,
                        &var_of,
                        &const_name,
                        &self_name,
                        &self_ty,
                    )?);
                    drop_stmts.push(i);
                }
            }
        }
        track_def(st, &mut defs, &mut tainted);
    }

    // Pass 2: sputs to the ACC_ENUM fields (direct new or intermediate).
    // Local sputs resolve through the ROLLING def map (updated at each
    // def-site statement) — a reused `v` must bind to the constant
    // defined most recently, not to the var's first registration.
    let mut defs2: HashMap<u32, &Expr> = HashMap::default();
    let mut tainted2 = false;
    let mut cur_def: HashMap<u32, usize> = HashMap::default();
    for (i, st) in match &body.body {
        Stmt::Block(v) => v.iter().enumerate(),
        _ => return None,
    } {
        if let Some((var, idx)) = def_site.get(&i) {
            cur_def.insert(*var, *idx);
        }
        if let Stmt::ExprStmt(Expr::Assign { target, value, .. }) = st {
            if let Expr::Field {
                cls,
                name: fname,
                is_static: true,
                ..
            } = &**target
            {
                if cls.as_ref() != class.name {
                    continue;
                }
                if !const_fields.iter().any(|f| f.name.as_str() == &**fname) {
                    continue;
                }
                if const_field.iter().any(|f| !f.is_empty() && f == &**fname) { { return None; } // duplicate assignment
                }
                let idx = match &**value {
                    Expr::Local { var, .. } => cur_def.get(var).copied()?,
                    Expr::New { cls: ncls, args, .. }
                        if ncls.as_ref() == class.name && args.len() >= 2 =>
                    {
                        if let (Expr::Const(ConstVal::Str(n)), Expr::Const(ConstVal::Int(ord0))) =
                            (&args[0], &args[1])
                        {
                            if java_ident(n).as_ref() != &**n || n.is_empty() { return None; }
                            let idx = const_name.len();
                            const_ord.push(*ord0 as i64);
                            const_name.push(n.to_string());
                            const_field.push(String::new());
                            const_extra.push(resolve_enum_extras(
                                &args[2..],
                                &defs2,
                                tainted2,
                                &var_of,
                                &const_name,
                                &self_name,
                                &self_ty,
                            )?);
                            idx
                        } else {
                            { return None; }
                        }
                    }
                    _ => return None,
                };
                const_field[idx] = fname.to_string();
                drop_stmts.push(i);
            }
        }
        track_def(st, &mut defs2, &mut tainted2);
    }
    // Own the reaching-def values: defs2 borrows body.body, and the
    // synthetic-constant scan below mutates the $VALUES array in place.
    let defs2_owned: std::collections::HashMap<u32, Expr> = defs2
        .iter()
        .map(|(k, v)| (*k, (*v).clone()))
        .collect();
    drop(defs2);
    let defs2: std::collections::HashMap<u32, &Expr> =
        defs2_owned.iter().map(|(k, v)| (*k, v)).collect();

    // Every ACC_ENUM field bound, every intermediate matched.
    // Every ACC_ENUM FIELD must be bound to a constant. The converse
    // need not hold: R8 drops the FIELD of a constant nothing reads
    // externally while the $VALUES array keeps it (weixin lite/api/n's
    // ON_DESTROY — a pass-1 local with no pass-2 sput). Field-less
    // constants keep their ctor-string source name.
    if const_field.iter().filter(|f| !f.is_empty()).count() != const_fields.len() {
               return None;
    }
    // Obfuscated enums rename the ACC_ENUM FIELD (d/e/f) while the ctor's
    // name STRING keeps the source identifier — the promoted constant
    // must be declared under the FIELD name: every reference in the
    // pool resolves through it. Using the name string broke all
    // cross-file references (weixin +2.9k when newly-promoted enums
    // swapped d/e/f for TEXT_ENTER_EDITING/…).
    for i in 0..const_name.len() {
        let field = &const_field[i];
        if !field.is_empty() && field != &const_name[i] {
            let id = java_ident(field);
            if id.is_empty() || id.as_ref() != field.as_str() {
                { return None; }
            }
            const_name[i] = field.clone();
        }
    }
    {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::default();
        if !const_name.iter().all(|c| seen.insert(c.as_str())) { return None; }
    }

    // Synthetic constants: entries built INLINE inside the $VALUES
    // array (`new Self("NAME", ord, ..)` with no ACC_ENUM field — R8
    // drops the field when nothing outside reads it; weixin u2/i's
    // "CONSTANT"). Promote them so the header declares the constant:
    // the raw `new` in the array is "无法实例化枚举类", and every
    // constant after the gap carries a SHIFTED ordinal (e was 2 in the
    // dex, renders as position 1).
    let mut synth: Vec<(i64, String, Vec<Expr>)> = Vec::new();
    {
        let self_cls: &str = class.name.as_str();
        if let Stmt::Block(vs) = &mut body.body {
            for st in vs.iter_mut() {
                crate::passes::walk_stmt_exprs(st, &mut |e| {
                    crate::passes::deep_rewrite(e, &mut |x| {
                        let Expr::NewArray { elem, init: Some(list), .. } = x else {
                            return;
                        };
                        if elem.erased() != JavaType::Object(self_cls.into()) {
                            return;
                        }
                        for slot in list.iter_mut() {
                            let (n, ord, rest) = match slot {
                                Expr::New { cls, args, raw: false, .. }
                                    if cls.as_ref() == self_cls && args.len() >= 2 =>
                                {
                                    match (&args[0], &args[1]) {
                                        (
                                            Expr::Const(ConstVal::Str(sv)),
                                            Expr::Const(ConstVal::Int(o)),
                                        ) => (sv.to_string(), *o as i64, args[2..].to_vec()),
                                        _ => continue,
                                    }
                                }
                                _ => continue,
                            };
                            if java_ident(&n).as_ref() != n.as_str() || n.is_empty() {
                                continue;
                            }
                            if const_name.contains(&n)
                                || synth.iter().any(|(_, s2, _)| *s2 == n)
                            {
                                continue;
                            }
                            let Some(extras) = resolve_enum_extras(
                                &rest,
                                &defs2,
                                tainted2,
                                &var_of,
                                &const_name,
                                &self_name,
                                &self_ty,
                            ) else {
                                continue;
                            };
                            *slot = Expr::Field {
                                owner: None,
                                cls: self_name.clone(),
                                name: std::sync::Arc::from(n.as_str()),
                                ty: self_ty.clone(),
                                is_static: true,
                            };
                            synth.push((ord, n, extras));
                        }
                    });
                });
            }
        }
    }

    // Pass 3: rewrite references to the intermediate locals into the
    // constant identifiers (static-field reads on Self). POSITION-AWARE:
    // a REUSED local must read as the constant defined most recently at
    // that statement — a var-level map made every `arr[k] = v` in the
    // $VALUES build reference the LAST constant (weixin +2.9k when this
    // regressed values() contents).
    {
        let mut cur: HashMap<u32, usize> = HashMap::default();
        if let Stmt::Block(vs) = &mut body.body {
            for (i, st) in vs.iter_mut().enumerate() {
                if let Some((var, idx)) = def_site.get(&i) {
                    cur.insert(*var, *idx);
                }
                crate::passes::walk_stmt_exprs(st, &mut |e| {
                    crate::passes::deep_rewrite(e, &mut |x| {
                        if let Expr::Local { var, .. } = x {
                            if let Some(&idx) = cur.get(var) {
                                let name: std::sync::Arc<str> =
                                    std::sync::Arc::from(const_name[idx].as_str());
                                *x = Expr::Field {
                                    owner: None,
                                    cls: self_name.clone(),
                                    name,
                                    ty: self_ty.clone(),
                                    is_static: true,
                                };
                            }
                        }
                    });
                });
            }
        }
    }

    // Pass 4: drop the definitions and their sputs.
    if let Stmt::Block(vs) = &mut body.body {
        let drop_set: std::collections::HashSet<usize> =
            drop_stmts.iter().copied().collect();
        vs.retain(|_| true);
        let mut idx = 0usize;
        vs.retain(|_| {
            let keep = !drop_set.contains(&idx);
            idx += 1;
            keep
        });
    }
    // The resolved args consumed the clinit locals' readers; prune the
    // now-dead defs from the remnant (impure inits survive as bare
    // expression statements — drop_dead_locals' standard contract).
    crate::passes::drop_dead_locals(&mut body.body);

    let mut merged: Vec<(i64, EnumConst)> = const_ord
        .into_iter()
        .zip(
            const_field
                .into_iter()
                .zip(const_name)
                .zip(const_extra)
                .map(|((field, name), extra_args)| EnumConst {
                    field,
                    name,
                    extra_args,
                }),
        )
        .collect();
    for (ord, name, extra_args) in synth {
        merged.push((
            ord,
            EnumConst {
                field: name.clone(),
                name,
                extra_args,
            },
        ));
    }
    merged.sort_by_key(|(o, _)| *o);
    // Ordinals must be UNIQUE and ASCENDING; gaps are legal — R8 drops
    // unused constants wholesale (weixin lite/api/n starts at ordinal
    // 1). Java derives ordinal() from declaration position, so a gap
    // shifts every later constant. Pad each gap with a synthetic
    // constant (`_r<k>`) to keep positions faithful — only when no
    // constant carries extra ctor args (a pad cannot invent them).
    // The synthetic values()/valueOf() the true-enum render suppresses
    // then see the pad (values() drift vs the dex $VALUES array — the
    // compilable-and-ordinal-faithful side of the trade).
    let has_extras = merged.iter().any(|(_, c)| !c.extra_args.is_empty());
    let mut member_taken: jdc_core::FxHashSet<String> = merged
        .iter()
        .map(|(_, c)| c.name.clone())
        .chain(
            class
                .static_fields
                .iter()
                .chain(class.instance_fields.iter())
                .map(|f| f.name.to_string()),
        )
        .chain(class.all_methods().map(|m| m.name.to_string()))
        .collect();
    let mut padded: Vec<EnumConst> = Vec::with_capacity(merged.len());
    let mut expect: i64 = 0;
    let mut pad_k = 0u32;
    for (ord, c) in merged {
        if ord < expect {
            { return None; } // duplicate/colliding ordinals — unfaithful
        }
        if ord > expect {
            if has_extras {
                { return None; } // cannot synthesize the missing ctor args
            }
            while expect < ord {
                let pname = loop {
                    let cand = format!("_r{pad_k}");
                    pad_k += 1;
                    if !member_taken.contains(&cand) {
                        member_taken.insert(cand.clone());
                        break cand;
                    }
                };
                padded.push(EnumConst {
                    field: pname.clone(),
                    name: pname,
                    extra_args: Vec::new(),
                });
                expect += 1;
            }
        }
        expect = ord + 1;
        padded.push(c);
    }
    let out = padded;
    Some((out, body))
}

/// Update the rolling reaching-def map for enum-arg resolution. Only
/// flat single-target definitions keep the tracking sound; any control
/// flow taints it (a linear scan can no longer prove WHICH definition
/// reaches later capture sites).
fn track_def<'e>(
    st: &'e Stmt,
    defs: &mut std::collections::HashMap<u32, &'e Expr>,
    tainted: &mut bool,
) {
    match st {
        Stmt::LocalDef { var, init, .. } => match init {
            Some(e) => {
                defs.insert(*var, e);
            }
            None => {
                defs.remove(var);
            }
        },
        Stmt::ExprStmt(Expr::Assign { target, value, op, .. }) => {
            if let Expr::Local { var, .. } = &**target {
                if matches!(op, jdc_core::ir::expr::AssignOp::Plain) {
                    defs.insert(*var, value);
                } else {
                    defs.remove(var);
                }
            }
        }
        // Plain expression statements (calls) don't define locals.
        Stmt::ExprStmt(_) => {}
        _ => *tainted = true,
    }
}

/// Resolve every enum-ctor extra arg to a self-contained expression, or
/// reject the enum mode (None). R8 reuses ONE register across all
/// constant constructions — revenuecat's LogIntent builds 11 of 12
/// emoji lists through the same `list` local, reassigned between the
/// `new Self(.., list)` sites — and enum constants render OUTSIDE the
/// clinit where no local is in scope: an unresolved `Local` used to
/// print as the vt-dummy name (`DEBUG(var0)` — per-constant
/// cannot-find). Each Local is replaced by its reaching pure definition
/// (recursively) or, when it names an earlier constant's intermediate,
/// by a static-field reference to that constant. Any expression shape
/// WITHOUT locals passes through untouched (enum args may be arbitrary
/// expressions — BinOp/Cast/Method/New — each renders exactly once per
/// constant, so no duplication concern applies). Only an unresolvable
/// Local (missing/tainted def, depth > 4) rejects the whole enum
/// detection; the class then falls back to plain-field rendering, which
/// always compiles.
fn resolve_enum_extras(
    extras: &[Expr],
    defs: &std::collections::HashMap<u32, &Expr>,
    tainted: bool,
    var_of: &std::collections::HashMap<u32, usize>,
    const_name: &[String],
    self_name: &std::sync::Arc<str>,
    self_ty: &jdc_core::ir::expr::TypeRef,
) -> Option<Vec<Expr>> {
    extras
        .iter()
        .map(|e| {
            let mut out = e.clone();
            let mut fail = false;
            resolve_locals_in(
                &mut out, defs, tainted, var_of, const_name, self_name, self_ty, 0,
                &mut fail,
            );
            if fail {
                None
            } else {
                Some(out)
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn resolve_locals_in(
    e: &mut Expr,
    defs: &std::collections::HashMap<u32, &Expr>,
    tainted: bool,
    var_of: &std::collections::HashMap<u32, usize>,
    const_name: &[String],
    self_name: &std::sync::Arc<str>,
    self_ty: &jdc_core::ir::expr::TypeRef,
    depth: u32,
    fail: &mut bool,
) {
    crate::passes::deep_rewrite(e, &mut |x| {
        if let Expr::Local { var, .. } = x {
            // An earlier constant's intermediate: a static-field read of
            // that constant (declared above — backward reference, legal
            // in enum ctor args).
            if let Some(&ci) = var_of.get(var) {
                if let Some(nm) = const_name.get(ci) {
                    *x = Expr::Field {
                        owner: None,
                        cls: self_name.clone(),
                        name: std::sync::Arc::from(nm.as_str()),
                        ty: self_ty.clone(),
                        is_static: true,
                    };
                    return;
                }
            }
            if tainted || depth > 4 {
                *fail = true;
                return;
            }
            let Some(d) = defs.get(var).copied() else {
                *fail = true;
                return;
            };
            let mut sub = d.clone();
            resolve_locals_in(
                &mut sub, defs, tainted, var_of, const_name, self_name, self_ty,
                depth + 1, fail,
            );
            if *fail {
                return;
            }
            *x = sub;
        }
    });
}

/// Render enum-constant constructor arguments via the shared expression
/// emitter (the VarTable is irrelevant for argument printing — no local
/// names appear — but the API requires one).
fn render_enum_args(ctx: &DexCtx<'_>, pool: &DexPool, args: &[Expr], out: &mut String) {
    let dummy_vt = jdc_core::var::VarTable::default();
    let mut p = Printer::new(ctx, &dummy_vt);
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        p.expr(a, 0, out);
    }
    let _ = pool;
}

#[allow(clippy::only_used_in_recursion)]
fn emit_class_body(
    pool: &DexPool,
    class: &PoolClass,
    ctx: &DexCtx<'_>,
    opts: &ClassOptions,
    out: &mut String,
    depth: usize,
) -> anyhow::Result<()> {
    let ind = indent(depth);
    // The class's own header must use the RENAMED display name — the
    // ctor path (is_init) already renames; a header declaring `class a`
    // while the file/ctor say `a_2` leaves the ctor looking like a
    // method with no return type (MinisApp's Y2.a vs y2.a package
    // case-collision: 2,518 javac parse errors).
    let cname = crate::apply_class_rename(&class.name);
    let (_, simple) = split_name(&cname);
    // A class emitted as its OWN top-level file (depth 0) must declare a
    // flat `$` name — `class Outer.Inner` is not declarable at file
    // scope. An INLINE nested member (depth > 0) declares its own
    // segment inside the parent's body.
    let simple = if depth == 0 {
        simple.to_string()
    } else {
        // R8 names can END in `$`: an empty last `$`-segment would
        // render `class  {` — keep the whole simple name.
        let seg = simple.rsplit('$').next().unwrap_or(&simple);
        if seg.is_empty() {
            simple.to_string()
        } else {
            seg.to_string()
        }
    };
    // A class NAMED `var`-style (taobao ships `tb.var`) cannot be
    // declared — restricted contextual type names escape here and in
    // the ctor, file name, and every type reference.
    let simple = if is_restricted_type_name(&simple) {
        format!("_{simple}")
    } else {
        simple
    };
    let is_iface = class.is_interface();
    let is_enum = class.is_enum();

    let mut head = String::new();
    let a = class.access;
    if a & ACC_PUBLIC != 0 {
        head.push_str("public ");
    }
    // Inline nested members need their `static` (interfaces/annotations
    // are implicitly static; a missing `static` on a member class makes
    // every `new Report(...)` site an "outer instance required" error).
    if depth > 0 && !is_iface && ctx.nested_is_static(&class.name) {
        head.push_str("static ");
    }
    if a & ACC_FINAL != 0 && !is_enum {
        head.push_str("final ");
    }
    if a & ACC_ABSTRACT != 0 && !is_iface {
        head.push_str("abstract ");
    }
    if a & ACC_ANNOTATION != 0 {
        head.push('@');
    }
    let mut enum_consts: Option<(Vec<EnumConst>, crate::method::MethodBody)> = if is_enum {
        collect_enum_constants(pool, class, ctx)
    } else {
        None
    };
    if let Some(ecs) = &enum_consts {
        // True `enum` declaration: constants render in the header, the
        // desugared boilerplate (const fields, their <clinit> inits —
        // stripped by strip_enum_const_inits — and the ACC_ENUM flags)
        // disappears. R8-renamed values()/valueOf() stay (they do not
        // collide with the compiler-generated ones); javac-named ones
        // are skipped at the method loop below.
        head.push_str("enum ");
        head.push_str(&sanitize_ref(&simple));
        let _ = ecs;
    } else if is_enum {
        // An enum with constant-specific bodies carries ACC_ABSTRACT —
        // `abstract final` is an illegal modifier combination; abstract
        // (already pushed above) suppresses the hardcoded final.
        let abstract_ = a & ACC_ABSTRACT != 0;
        head.push_str(if abstract_ {
            "/* enum */ class "
        } else {
            "/* enum */ final class "
        });
        head.push_str(&sanitize_ref(&simple));
    } else if is_iface {
        head.push_str("interface ");
        head.push_str(&sanitize_ref(&simple));
    } else {
        head.push_str("class ");
        head.push_str(&sanitize_ref(&simple));
    }
    // The obscured-super import was emitted at the package line; the
    // clause must render the SIMPLE name (the qualified form binds to
    // the class itself even with the import present).
    let render_super = |sup: &String| -> String { print_class_name(pool, sup) };
    if is_iface {
        // A @interface cannot declare extends at all (JLS 9.6): the dex
        // interface table lists java/lang/annotation/Annotation for
        // every annotation type — rendering it produced "对于
        // @interfaces, 不允许 'extends'" (weibo 1100).
        let is_annot = a & ACC_ANNOTATION != 0;
        if !class.interfaces.is_empty() && !is_annot {
            head.push_str(" extends ");
            head.push_str(&join_dotted(pool, &class.interfaces));
        }
    } else {
        if let Some(sup) = &class.super_name {
            if sup != "java/lang/Object" && !is_enum {
                head.push_str(" extends ");
                head.push_str(&render_super(sup));
            }
        }
        if !class.interfaces.is_empty() {
            head.push_str(if is_iface {
                " extends "
            } else {
                " implements "
            });
            head.push_str(&join_dotted(pool, &class.interfaces));
        }
    }

    out.push_str(&ind);
    out.push_str(&head);
    out.push_str(" {\n");

    // True-enum constant list: the constants lead the body, ahead of
    // any remaining fields.
    if let Some((ecs, _)) = &enum_consts {
        for (i, ec) in ecs.iter().enumerate() {
            out.push_str(&indent(depth + 1));
            out.push_str(&java_ident(&ec.name));
            if !ec.extra_args.is_empty() {
                let mut line = String::from("(");
                render_enum_args(ctx, pool, &ec.extra_args, &mut line);
                line.push(')');
                out.push_str(&line);
            }
            if i + 1 < ecs.len() {
                out.push_str(",\n");
            } else {
                out.push_str(";\n");
            }
        }
        out.push('\n');
    }

    // Fields.
    // Kotlin `object`/lazy singletons: the dex static value for INSTANCE
    // is null while the clinit assigns it — rendering `= null` as the
    // field initializer made the clinit assignment illegal ("无法为
    // static final 变量 INSTANCE 分配值", okio SegmentPool family). A
    // static final whose static value is null AND which the clinit
    // sputs renders as a blank final. Scan the clinit's raw SPuts once.
    let clinit_sputs: Option<jdc_core::FxHashSet<(std::sync::Arc<str>, std::sync::Arc<str>)>> =
        // Gate: only classes with a null-valued final static can produce
        // a blank final — skip the clinit decode otherwise (decoding it
        // for every class cost seconds on 98k-class corpora).
        (|| {
            let has_null_final = class.static_fields.iter().enumerate().any(|(i, f)| {
                f.access & crate::access::ACC_FINAL != 0
                    && matches!(class.static_values.get(i), Some(StaticValue::Null))
            });
            if !has_null_final {
                return None;
            }
            let m = class.all_methods().find(|m| &*m.name == "<clinit>")?;
            let dex = pool.dex(m.dex_idx)?;
            let ci = dex.code_at(m.code_off)?;
            let mut set: jdc_core::FxHashSet<(
                std::sync::Arc<str>,
                std::sync::Arc<str>,
            )> = jdc_core::FxHashSet::default();
            for ins in &ci.insns {
                if let InsnKind::SPut { field_idx, .. } = ins.kind {
                    let fid = dex.field(field_idx);
                    // clinit only writes own-class fields here; match
                    // (name, type) against the field list below.
                    set.insert((
                        std::sync::Arc::from(dex.string(fid.name_idx)),
                        std::sync::Arc::from(dex.type_name(fid.type_idx)),
                    ));
                }
            }
            Some(set)
        })();
    let mut field_emitted = false;
    for (i, f) in class.static_fields.iter().enumerate() {
        // Enum constant fields became the header list above.
        if enum_consts.is_some() && f.access & crate::access::ACC_ENUM != 0 {
            continue;
        }
        if field_emitted || !class.instance_fields.is_empty() {
            out.push('\n');
        }
        field_emitted = true;
        // A blank-final: static final, null static value, clinit-assigned.
        let blank_final = f.access & crate::access::ACC_FINAL != 0
            && matches!(class.static_values.get(i), Some(StaticValue::Null))
            && clinit_sputs
                .as_ref()
                .is_some_and(|s| {
                    s.contains(&(
                        std::sync::Arc::from(f.name.as_str()),
                        std::sync::Arc::from(f.desc.as_str()),
                    ))
                });
        emit_field(
            pool,
            f,
            if blank_final {
                None
            } else {
                class.static_values.get(i)
            },
            &class.name,
            out,
            depth + 1,
            true,
            class.is_interface(),
        );
    }
    if !class.instance_fields.is_empty() && !class.static_fields.is_empty() {
        out.push('\n');
    }
    for (i, f) in class.instance_fields.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        emit_field(pool, f, None, &class.name, out, depth + 1, false, false);
    }

    // Methods.
    // Duplicated (name, parameter-types) pairs in one class are javac
    // "method is already defined" errors (weibo: 11.5k hits). Source
    // cannot declare them, so exactly one survives: covariant bridges
    // (`Object get(int)` beside `ByteString get(int)` — the compiler
    // GENERATES bridges), plus R8 output that lost the bridge flag
    // (okio Buffer.clone, interface getView re-declarations). A
    // non-bridge method outranks a bridge for the same signature
    // (bridge bodies are delegation stubs); first occurrence otherwise.
    // Never drop a signature outright — the last method standing for a
    // key is always rendered.
    fn sig_key(m: &PoolMethod) -> (&str, &str) {
        let d: &str = &m.desc;
        let lo = d.find('(').map(|i| i + 1).unwrap_or(0);
        let hi = d.find(')').unwrap_or(d.len());
        (&*m.name, &d[lo..hi])
    }
    let methods: Vec<&PoolMethod> = class.all_methods().collect();
    let mut claim: jdc_core::FxHashMap<(&str, &str), usize> = jdc_core::FxHashMap::default();
    for (i, m) in methods.iter().enumerate() {
        let key = sig_key(m);
        match claim.get(&key) {
            Some(&j)
                if m.access & crate::access::ACC_BRIDGE == 0
                    && methods[j].access & crate::access::ACC_BRIDGE != 0 =>
            {
                claim.insert(key, i);
            }
            None => {
                claim.insert(key, i);
            }
            _ => {}
        }
    }
    // NOTE: a same-signature bridge is already retired by the claim map
    // above. An ERASURE-shaped bridge (SAM/variance: params are the
    // Object-erased types of a sibling's) must NOT be skipped: with raw
    // (non-generic) interface rendering the bridge is exactly what
    // satisfies the interface — skipping it turned every Kotlin lambda
    // class into "不是抽象的, 并且未覆盖…invoke(Object,Object)" (reqable
    // +17), while the "对invoke的引用不明确" it appeared to fix was a
    // missing-kotlin-classpath cascade, not a ddc bug.
    let mut emitted_any = !class.static_fields.is_empty() || !class.instance_fields.is_empty();
    for (i, m) in methods.iter().enumerate() {
        if &*m.name == "<clinit>" {
            continue; // rendered after the fields
        }
        if claim.get(&sig_key(m)) != Some(&i) {
            continue;
        }
        // True-enum rendering: javac auto-generates values()/valueOf()
        // — the original-named ones must not re-declare (R8-renamed
        // copies stay, they do not collide).
        if let Some((ecs, _)) = &enum_consts {
            let d = m.parsed_desc();
            let self_arr = d.as_ref().map(|d| d.ret == JavaType::Array(Box::new(JavaType::Object(class.name.as_str().into()))));
            let self_ret = d.as_ref().map(|d| d.ret == JavaType::Object(class.name.as_str().into()));
            let no_args = d.as_ref().map(|d| d.args.is_empty()).unwrap_or(false);
            let one_str = d
                .as_ref()
                .map(|d| d.args.len() == 1 && matches!(d.args[0], JavaType::Object(ref n) if n.as_ref() == "java/lang/String"))
                .unwrap_or(false);
            let static_ = m.is_static();
            if static_ && no_args && self_arr == Some(true) && &*m.name == "values" {
                continue;
            }
            if static_ && one_str && self_ret == Some(true) && &*m.name == "valueOf" {
                continue;
            }
            // Enum constructors must be private in source form.
            let m_owned: Option<PoolMethod> = if &*m.name == "<init>" {
                let mut c = (*m).clone();
                c.access = (c.access & !(crate::access::ACC_PUBLIC | crate::access::ACC_PROTECTED | crate::access::ACC_PRIVATE)) | crate::access::ACC_PRIVATE;
                Some(c)
            } else {
                None
            };
            let m_ref: &PoolMethod = m_owned.as_ref().unwrap_or(m);
            let mark = out.len();
            if emitted_any {
                out.push('\n');
            }
            if emit_method(pool, class, ctx, m_ref, depth + 1, true, out)? {
                emitted_any = true;
            } else {
                out.truncate(mark);
            }
            let _ = ecs;
            continue;
        }
        let mark = out.len();
        if emitted_any {
            out.push('\n');
        }
        if emit_method(pool, class, ctx, m, depth + 1, enum_consts.is_some(), out)? {
            emitted_any = true;
        } else {
            out.truncate(mark);
        }
    }
    // Inherited-ctor bridges: dex method refs resolve through the
    // hierarchy, so `new C(args)` against a class C with NO declared
    // <init> legally targets the SUPERCLASS ctor (weixin tenpay
    // `new m(map)` — m declares nothing, i.<init>(HashMap) does). Java
    // has no inherited constructors: without a bridge, C's implicit
    // default ctor is the only one and every arg-carrying construction
    // fails ("无法将类 m中的构造器 m应用到给定类型; 需要: 没有参数").
    // Mirror the nearest ctor-declaring ancestor's public/protected
    // ctors as thin `super(..)` delegations.
    if !class.is_interface()
        && enum_consts.is_none()
        && !class.all_methods().any(|m| &*m.name == "<init>")
    {
        if let Some(mut sup) = class.super_name.clone() {
            loop {
                if sup == "java/lang/Object" {
                    break;
                }
                let Some(sc) = pool.get(&sup) else { break };
                let ctors: Vec<&PoolMethod> = sc
                    .all_methods()
                    .filter(|m| &*m.name == "<init>")
                    .collect();
                if !ctors.is_empty() {
                    for sm in ctors {
                        if sm.access & crate::access::ACC_PRIVATE != 0
                            || (sm.access
                                & (crate::access::ACC_PUBLIC | crate::access::ACC_PROTECTED)
                                == 0)
                        {
                            continue; // private/package-private: no legal bridge
                        }
                        let Some(d) = sm.parsed_desc() else { continue };
                        let mods = if sm.access & crate::access::ACC_PUBLIC != 0 {
                            "public "
                        } else {
                            "protected "
                        };
                        if emitted_any {
                            out.push('\n');
                        }
                        out.push_str(&format!("    {}", "    ".repeat(depth)));
                        out.push_str(mods);
                        out.push_str(&sanitize_ref(&simple));
                        out.push('(');
                        let mut names = Vec::with_capacity(d.args.len());
                        for (i, a) in d.args.iter().enumerate() {
                            if i > 0 {
                                out.push_str(", ");
                            }
                            out.push_str(&type_name(pool, a));
                            let nm = format!("p{}", i + 1);
                            out.push(' ');
                            out.push_str(&nm);
                            names.push(nm);
                        }
                        out.push_str(") {\n");
                        out.push_str(&format!("    {}    super({});\n", "    ".repeat(depth), names.join(", ")));
                        out.push_str(&format!("    {}}}\n", "    ".repeat(depth)));
                        emitted_any = true;
                    }
                    break;
                }
                match &sc.super_name {
                    Some(n) => sup = n.clone(),
                    None => break,
                }
            }
        }
    }

    // Static initializer. INTERFACES cannot carry a `static { }` block in
    // Java — their clinit only assigns constants, which static_values (or
    // the `= null` default) already render as field initializers; skip
    // the block entirely.
    let skip_clinit = class.is_interface();
    if let Some((_, clinit_body)) = enum_consts.take() {
        // True-enum <clinit>: the constant assignments are already gone;
        // render the remainder (the $VALUES array build) directly.
        // Skip an empty remainder (all statements were constant inits).
        let empty = match &clinit_body.body {
            Stmt::Block(v) => v.iter().all(|s| matches!(s, Stmt::Block(b) if b.is_empty())),
            _ => false,
        };
        if !empty {
            if emitted_any {
                out.push('\n');
            }
            out.push_str(&indent(depth + 1));
            out.push_str("static {\n");
            let p = Printer::new(ctx, &clinit_body.vt);
            let text = p.with_indent(depth + 2).into_string(&clinit_body.body);
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                out.push_str(line);
                out.push('\n');
            }
            out.push_str(&indent(depth + 1));
            out.push_str("}\n");
            emitted_any = true;
        }
    } else if let Some(clinit) = (!skip_clinit)
        .then(|| class.all_methods().find(|m| &*m.name == "<clinit>"))
        .flatten()
    {
        let mark = out.len();
        if emitted_any {
            out.push('\n');
        }
        if emit_method(pool, class, ctx, clinit, depth + 1, false, out)? {
            emitted_any = true;
        } else {
            out.truncate(mark);
        }
    }

    // Nested member classes (clean `$` tails only; anonymous/local/lambda
    // classes are emitted as their own top-level files by the driver).
    for nested in nested_members(pool, class, ctx) {
        if emitted_any {
            out.push('\n');
        }
        let nested_ctx = DexCtx::new(pool, nested);
        out.push('\n');
        emit_class_body(pool, nested, &nested_ctx, opts, out, depth + 1)?;
        emitted_any = true;
    }

    out.push_str(&ind);
    out.push_str("}\n");
    Ok(())
}

/// Member classes of `class` (direct children by the `$` chain / member
/// annotations), excluding anonymous / local / lambda shapes.
fn nested_members<'a>(
    pool: &'a DexPool,
    class: &'a PoolClass,
    ctx: &DexCtx<'_>,
) -> Vec<&'a PoolClass> {
    let mut out = Vec::new();
    for name in pool.children_of(&class.name) {
        let Some(pc) = pool.get(name) else { continue };
        // Only a REAL `outer$tail` name is an inline member: the
        // children index also carries annotation-derived outers
        // (EnclosingClass) with no naming relationship to the child
        // (obfuscated apps pair a 1-char name with a long enclosing
        // descriptor) — the blind slice panicked there (alipay
        // `a.a.a.a.c`, exposed once children_of actually returned data).
        let Some(rest) = name
            .strip_prefix(class.name.as_str())
            .and_then(|t| t.strip_prefix('$'))
        else {
            continue;
        };
        if rest.is_empty() || rest.starts_with('-') {
            continue;
        }
        if ctx.find_outer(name).as_deref() != Some(class.name.as_str()) {
            continue;
        }
        let tail = rest.rsplit('$').next().unwrap_or(rest);
        if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) {
            continue; // anonymous
        }
        if tail.starts_with(|c: char| c.is_ascii_digit()) {
            continue; // local
        }
        out.push(pc);
    }
    out
}

// Eight parameters are all load-bearing (pool, field, static value,
// owner class, output, depth, staticness, interface-init requirement);
// bundling them into a struct would obscure each call site.
#[allow(clippy::too_many_arguments)]
fn emit_field(
    pool: &DexPool,
    f: &crate::PoolField,
    init: Option<&StaticValue>,
    f_class: &str,
    out: &mut String,
    depth: usize,
    is_static: bool,
    require_init: bool,
) {
    let ind = indent(depth);
    let mut line = String::new();
    let a = f.access;
    if a & ACC_PUBLIC != 0 {
        line.push_str("public ");
    } else if a & ACC_PRIVATE != 0 {
        line.push_str("private ");
    } else if a & ACC_PROTECTED != 0 {
        line.push_str("protected ");
    }
    if is_static {
        line.push_str("static ");
    }
    if a & ACC_FINAL != 0 {
        line.push_str("final ");
    }
    if a & ACC_SYNTHETIC != 0 {
        line.push_str("/* synthetic */ ");
    }
    if a & ACC_TRANSIENT_HINT != 0 {
        line.push_str("transient ");
    }
    if a & ACC_VOLATILE_HINT != 0 {
        line.push_str("volatile ");
    }
    let ty = type_name(pool, &desc_type(&f.desc));
    line.push_str(&ty);
    line.push(' ');
    let fname = jdc_core::rename::field_display(f_class, &f.name, &f.desc).unwrap_or(&*f.name);
    line.push_str(&java_ident(fname));
    let mut rendered = None;
    if let Some(v) = init {
        rendered = render_static_value(pool, v, f_class);
    }
    if rendered.is_none() && require_init {
        // Interface fields MUST have an initializer in Java; the dex may
        // not carry a static_values entry for a compile-time-constant the
        // compiler folded away. Keep it compilable.
        let default = match desc_type(&f.desc) {
            JavaType::Boolean => "false",
            JavaType::Byte | JavaType::Short | JavaType::Char | JavaType::Int => "0",
            JavaType::Long => "0L",
            JavaType::Float => "0.0F",
            JavaType::Double => "0.0",
            _ => "null",
        };
        rendered = Some(default.to_string());
    }
    if let Some(text) = rendered {
        // A long field holding an `Int(i64)` static value needs the `L`
        // suffix: without it the literal is an int and overflows
        // (`long d = -6343169151696340687` failed javac).
        let text = if matches!(desc_type(&f.desc), JavaType::Long)
            && text.chars().all(|c| c.is_ascii_digit() || c == '-')
        {
            format!("{text}L")
        } else {
            text
        };
        line.push_str(" = ");
        line.push_str(&text);
    }
    line.push(';');
    out.push_str(&ind);
    out.push_str(&line);
    out.push('\n');
}

// DEX field access_flags bits 0x40/0x80 are bridge/varargs for METHODS and
// volatile(0x40)/transient(0x80) for fields.
pub const ACC_VOLATILE_HINT: u32 = 0x40;
pub const ACC_TRANSIENT_HINT: u32 = 0x80;

fn render_static_value(pool: &DexPool, v: &StaticValue, owner: &str) -> Option<String> {
    Some(match v {
        StaticValue::Int(i) => i.to_string(),
        StaticValue::Float(f) => format_float(*f as f64, true),
        StaticValue::Double(d) => format_float(*d, false),
        StaticValue::Str(s) => format!("\"{}\"", escape_string(s)),
        StaticValue::Type(t) => format!("{}.class", dotted(t)),
        StaticValue::Boolean(b) => b.to_string(),
        StaticValue::Null => "null".into(),
        StaticValue::Field(cls, name) => {
            let n = java_ident(name);
            if cls == owner {
                n.into_owned()
            } else {
                format!("{}.{}", print_class_name(pool, cls), n)
            }
        }
        StaticValue::Other => return None,
    })
}

/// Render one method straight into the class buffer `out`. Returns
/// whether anything was written (skips: no descriptor, deferred to a
/// monitored thread). The old shape returned a per-method String that
/// the caller copied in — one extra full copy of every method body.
/// `enum_promoted`: the class rendered as a true `enum` declaration
/// (constants in the header), which changes what a ctor signature may
/// declare.
fn emit_method(
    pool: &DexPool,
    class: &PoolClass,
    ctx: &DexCtx<'_>,
    m: &PoolMethod,
    depth: usize,
    enum_promoted: bool,
    out: &mut String,
) -> anyhow::Result<bool> {
    let ind = indent(depth);
    let desc = m.parsed_desc();

    // Signature.
    let mut sig = String::with_capacity(192);
    let a = m.access;
    if a & ACC_PUBLIC != 0 {
        sig.push_str("public ");
    } else if a & ACC_PRIVATE != 0 {
        sig.push_str("private ");
    } else if a & ACC_PROTECTED != 0 {
        sig.push_str("protected ");
    }
    let is_clinit = &*m.name == "<clinit>";
    let is_init = &*m.name == "<init>";
    if a & ACC_STATIC != 0 || is_clinit {
        sig.push_str("static ");
    }
    if a & ACC_FINAL != 0 {
        sig.push_str("final ");
    }
    // `abstract synchronized` is an illegal combination — obfuscated
    // builds mark abstract bridges synchronized (WhatsApp
    // SQLiteOpenHelper).
    if a & (ACC_SYNCHRONIZED | ACC_DECLARED_SYNCHRONIZED) != 0 && a & ACC_ABSTRACT == 0 {
        sig.push_str("synchronized ");
    }
    if a & ACC_NATIVE != 0 {
        sig.push_str("native ");
    }
    if a & ACC_ABSTRACT != 0 {
        sig.push_str("abstract ");
    }
    if a & ACC_SYNTHETIC != 0 {
        sig.push_str("/* synthetic */ ");
    }

    // Body (needed for parameter names even for abstract methods).
    let mut body = decompile_method(pool, class, m).ok().flatten();
    // An interface method WITH a body is a `default` method (JLS 9.4.3)
    // unless static/private — dex carries no `default` flag, so the
    // plain form rendered an abstract signature with a body and javac
    // rejected every one ("接口抽象方法不能带有主体", weixin 1138). A
    // default REQUIRES a body: only push it once the body is confirmed
    // (a failed decompile renders a body-less declaration).
    if class.is_interface()
        && a & (ACC_STATIC | ACC_ABSTRACT | ACC_NATIVE | ACC_PRIVATE | ACC_ANNOTATION) == 0
        && body.is_some()
    {
        sig.insert_str(0, "default ");
    }
    let param_names: Vec<String> = body
        .as_ref()
        .map(|b| {
            let mut ps: Vec<(u16, String)> =
                b.vt.vars
                    .iter()
                    .filter(|v| v.is_param && v.name != "this")
                    .map(|v| (v.slot, v.name.clone()))
                    .collect();
            ps.sort_by_key(|(s, _)| *s);
            ps.into_iter().map(|(_, n)| n).collect()
        })
        .unwrap_or_else(|| {
            (0..desc.as_ref().map(|d| d.args.len()).unwrap_or(0))
                .map(|i| format!("p{}", i + 1))
                .collect()
        });

    // Non-static member-inner ctor: the synthetic outer instance rides
    // as args[0] (typed as the direct enclosing class). Every emission
    // site already passes it implicitly — qualified `outer.new Inner(..)`,
    // `this.new Inner(..)` from inside the outer, `super(..)` after
    // skip_outer_arg — so the signature drops the param and the body's
    // references become `Outer.this`. Without this, every construction
    // site fails javac arity ("无法将类…构造器…应用到给定类型" — the
    // guava androidx inner-class families, d8 capture lambdas).
    let mut inner_arg0 = 0usize;
    if is_init {
        let tail = class.name.rsplit('$').next().unwrap_or("");
        let digit_simple = !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit());
        if let Some(d) = desc.as_ref() {
            if let Some(JavaType::Object(outer)) = d.args.first() {
                // The direct enclosing class: nesting annotation when
                // present, else the `$`-chain parent (find_outer_name —
                // d8 lambdas are `Outer$$ExternalSyntheticLambdaN`, where
                // a plain rsplit leaves a trailing `$`). It must agree
                // with the this$0 type for the rewrite to fire.
                let enclosing: Option<String> = class
                    .nesting
                    .enclosing_class
                    .clone()
                    .or_else(|| crate::find_outer_name(pool, &class.name));
                let direct = enclosing.as_deref() == Some(outer.as_ref());
                if direct && !digit_simple && ctx.class_has_this0(&class.name) {
                    let param0 = body.as_ref().and_then(|b| {
                        b.vt.vars
                            .iter()
                            .find(|v| v.is_param && v.name != "this")
                            .map(|v| v.id)
                    });
                    let written = match (&body, param0) {
                        (Some(b), Some(p0)) => crate::passes::local_is_written(&b.body, p0),
                        _ => false,
                    };
                    if let (Some(p0), false) = (param0, written) {
                        if let Some(b) = body.as_mut() {
                            // this()-delegations to the SAME class drop the
                            // outer arg (all of its ctors share the strip);
                            // a super target joins when it is an inner of
                            // the same enclosing family.
                            let mut eligible: Vec<String> = vec![class.name.clone()];
                            if let Some(sup) = &class.super_name {
                                let sup_enclosing =
                                    crate::find_outer_name(pool, sup);
                                if sup_enclosing.as_deref() == Some(outer.as_ref())
                                    && ctx.class_has_this0(sup)
                                {
                                    eligible.push(sup.clone());
                                }
                            }
                            let renamed = crate::apply_class_rename(outer);
                            let display = dotted(renamed.as_ref());
                            let ty =
                                jdc_core::ir::expr::TypeRef::J(JavaType::Object(outer.clone()));
                            crate::passes::rewrite_inner_ctor_outer_param(
                                &mut b.body,
                                p0,
                                &display,
                                &ty,
                                &eligible,
                            );
                            inner_arg0 = 1;
                        }
                    }
                }
            }
        }
    }

    if is_clinit {
        // `static { ... }` — the caller strips the method name/params.
    } else {
        let Some(d) = &desc else { return Ok(false) };
        if is_init {
            // The ctor name must equal the DECLARED class name of its
            // file: flat `$` at depth 0 (own file), own segment when
            // inlined in the parent at depth > 0. Case-renamed classes
            // use their display name.
            let dname = crate::apply_class_rename(&class.name);
            let (_, simple) = split_name(&dname);
            // emit_method's depth is the METHOD indent = class depth + 1:
            // own-file classes (depth 0 header → method depth 1) need the
            // flat `$` ctor name; inline nested members use their segment.
            let base = if depth <= 1 {
                simple.to_string()
            } else {
                // R8 names can END in `$` (`ThreadMsg$$$`): the last
                // `$`-segment is empty — keep the whole simple name.
                let seg = simple.rsplit('$').next().unwrap_or(&simple);
                if seg.is_empty() {
                    simple.to_string()
                } else {
                    seg.to_string()
                }
            };
            let name = if is_restricted_type_name(&base) {
                format!("_{base}")
            } else {
                base
            };
            sig.push_str(&java_ident(&name));
        } else {
            sig.push_str(&type_name(pool, &d.ret));
            sig.push(' ');
            let mname =
                jdc_core::rename::field_display(&class.name, &m.name, &m.desc).unwrap_or(&*m.name);
            sig.push_str(&java_ident(mname));
        }
        sig.push('(');
        // A promoted enum ctor's dex descriptor carries the compiler-
        // synthesized `(String name, int ordinal)` prefix — JLS forbids
        // declaring those (they are implicit in the `A(args)` constant
        // declarations the promotion emits, and strip_enum_ctor_super
        // already removed the `super(name, ordinal, ..)` delegation).
        // Skip the pair in the signature, or every constant declaration
        // fails javac arity ("无法将枚举…构造器…应用到给定类型"). Only
        // when the body never reads the two params — code that genuinely
        // uses them keeps the declared form.
        let mut arg0 = 0;
        if enum_promoted
            && is_init
            && matches!(d.args.first(), Some(JavaType::Object(s)) if s.as_ref() == "java/lang/String")
            && matches!(d.args.get(1), Some(JavaType::Int))
        {
            // The Kotlin default-arg bridge ctor reads name/ordinal in
            // its `this(str, p2, ..)` delegation only — the LEADING pair
            // of a this()-delegation drops together with the params, so
            // those reads do not block the strip (the constants' extra
            // args match the bridge's user params, not the 2-param user
            // ctor).
            let synthetic: Vec<u32> = body
                .as_ref()
                .map(|b| {
                    b.vt
                        .vars
                        .iter()
                        .filter(|v| v.is_param && v.name != "this")
                        .take(2)
                        .map(|v| v.id)
                        .collect()
                })
                .unwrap_or_default();
            // The delegation-leading extension only applies to the
            // Kotlin default-arg BRIDGE: its descriptor ends with
            // kotlin/jvm/internal/DefaultConstructorMarker. A REAL user
            // ctor whose first two params are (String, int) and merely
            // forwards them must keep its signature (weixin +2.9k when
            // ungated).
            let is_bridge = matches!(
                d.args.last(),
                Some(JavaType::Object(m)) if m.as_ref() == "kotlin/jvm/internal/DefaultConstructorMarker"
            );
            let mut lead = [0usize, 0usize];
            let refs_ok = body.as_ref().is_some_and(|b| {
                let uses = crate::passes::count_locals_stmts(std::slice::from_ref(&b.body));
                let mut c = b.body.clone();
                let cls_name = class.name.as_str();
                crate::passes::walk_stmt_exprs(&mut c, &mut |e| {
                    if let Expr::Method { name: mn, cls: mc, args, is_special, .. } = e {
                        if &**mn == "<init>" && *is_special && mc.as_ref() == cls_name {
                            for (k, a) in args.iter().enumerate() {
                                if let Expr::Local { var: v, .. } = a {
                                    if let Some(pi) = synthetic.iter().position(|p| p == v) {
                                        if k == pi && k < 2 {
                                            lead[pi] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
                synthetic.iter().enumerate().all(|(pi, id)| {
                    let total = uses.get(id).copied().unwrap_or(0);
                    if is_bridge {
                        total == lead[pi]
                    } else {
                        total == 0
                    }
                })
            });
            if refs_ok {
                arg0 = 2;
                // Drop the leading (name, ordinal) args of the this()
                // delegations.
                if let (Some(b), true) = (body.as_mut(), is_bridge) {
                    let cls_name = class.name.as_str();
                    let syn = synthetic;
                    crate::passes::walk_stmt_exprs(&mut b.body, &mut |e| {
                        if let Expr::Method { name: mn, cls: mc, args, is_special, .. } = e {
                            if &**mn == "<init>" && *is_special && mc.as_ref() == cls_name {
                                let mut drop_n = 0;
                                for a in args.iter().take(2) {
                                    if let Expr::Local { var: v, .. } = a {
                                        if syn.contains(v) && drop_n == v - syn[0] {
                                            drop_n += 1;
                                            continue;
                                        }
                                    }
                                    break;
                                }
                                for _ in 0..drop_n {
                                    args.remove(0);
                                }
                            }
                        }
                    });
                }
            }
        }
        let n = d.args.len();
        let varargs = a & ACC_VARARGS != 0 && n > 0;
        let arg0 = arg0 + inner_arg0;
        for (i, arg) in d.args.iter().enumerate().skip(arg0) {
            if i > arg0 {
                sig.push_str(", ");
            }
            let name = param_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("p{}", i));
            if varargs && i + 1 == n {
                if let JavaType::Array(inner) = arg {
                    sig.push_str(&type_name(pool, inner));
                    sig.push_str("...");
                } else {
                    sig.push_str(&type_name(pool, arg));
                }
            } else {
                sig.push_str(&type_name(pool, arg));
            }
            sig.push(' ');
            // Param names come from dex debug info — obfuscated apps
            // name them `_` (reserved since Java 9) or after keywords.
            sig.push_str(&java_ident(&name));
        }
        sig.push(')');
    }

    if is_clinit {
        let Some(b) = body else { return Ok(false) };
        // Direct render: the printer starts at the method's ABSOLUTE
        // indent and appends into the same buffer that carries the
        // header — no intermediate body string, no per-line re-indent
        // pass (two full copies of every method body saved).
        out.push_str(&ind);
        out.push_str("static {\n");
        let hdr = out.len();
        let printer = Printer::new(ctx, &b.vt)
            .with_indent(depth + 1)
            .with_output(std::mem::take(out));
        let t_print = std::time::Instant::now();
        let mut rendered = printer.into_string(&b.body);
        crate::method::phase_hit(3, t_print);
        if rendered[hdr..].trim().is_empty() {
            rendered.truncate(hdr);
        }
        rendered.push_str(&ind);
        rendered.push_str("}\n");
        *out = rendered;
        return Ok(true);
    }
    if a & (ACC_ABSTRACT | ACC_NATIVE) != 0 || body.is_none() {
        out.push_str(&ind);
        out.push_str(&sig);
        out.push_str(";\n");
        return Ok(true);
    }
    let Some(b) = body else { return Ok(false) };

    // Direct render at the absolute indent level (see the clinit path).
    let mut printer = Printer::new(ctx, &b.vt).with_indent(depth + 1);
    match &b.desc.ret {
        JavaType::Boolean => {
            printer = printer.with_ret_bool(true);
        }
        JavaType::Char => {
            printer = printer.with_ret_char(true);
        }
        JavaType::Byte => {
            printer = printer.with_ret_narrow(true, false);
        }
        JavaType::Short => {
            printer = printer.with_ret_narrow(false, true);
        }
        _ => {}
    }
    out.push_str(&ind);
    out.push_str(&sig);
    out.push_str(" {\n");
    let hdr = out.len();
    let printer = printer.with_output(std::mem::take(out));
    let t_print = std::time::Instant::now();
    let mut rendered = printer.into_string(&b.body);
    crate::method::phase_hit(3, t_print);
    if rendered[hdr..].trim().is_empty() {
        rendered.truncate(hdr);
    }
    rendered.push_str(&ind);
    rendered.push_str("}\n");
    *out = rendered;
    Ok(true)
}

// ---------------------------------------------------------------------------
// Naming helpers
// ---------------------------------------------------------------------------

fn indent(depth: usize) -> String {
    "    ".repeat(depth)
}

/// `$`-separated nesting rendered with dots — but ONLY when every
/// segment is a clean Java identifier (a genuine member class chain).
/// Anonymous (`Outer$1`), Kotlin synthetic (`Version$bigInteger$2`,
/// `...$$inlined$collect$1`) and local-class tails are NOT member
/// classes Java can name; the whole name stays flat with `$` (ddc emits
/// them as their own top-level files).
/// Kotlin emits method names like `invokeSuspend$lambda-0` — `-` (and
/// any other non-identifier character) is not legal Java. Deterministic
/// mapping, applied identically at declaration and call sites.
pub(crate) fn java_ident(name: &str) -> std::borrow::Cow<'_, str> {
    // env::var_os is an environ lock+scan — java_ident runs per IDENTIFIER
    // (millions per APK), where it profiled as __NSGetEnviron.
    static DBG_IDENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DBG_IDENT.get_or_init(|| std::env::var_os("DDC_DBG_IDENT").is_some()) {
        eprintln!("[ident] {name:?}");
    }
    let clean = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$');
    // Kotlin names fields/methods `default` (Companion.default) — no Java
    // program can declare or reference a keyword; the identical mapping
    // lives in jdc-core's call-site sanitizer.
    let keyword = matches!(
        name,
        "abstract"
            | "assert"
            | "boolean"
            | "break"
            | "byte"
            | "case"
            | "catch"
            | "char"
            | "class"
            | "const"
            | "continue"
            | "default"
            | "do"
            | "double"
            | "else"
            | "enum"
            | "extends"
            | "final"
            | "finally"
            | "float"
            | "for"
            | "goto"
            | "if"
            | "implements"
            | "import"
            | "instanceof"
            | "int"
            | "interface"
            | "long"
            | "native"
            | "new"
            | "package"
            | "private"
            | "protected"
            | "public"
            | "return"
            | "short"
            | "static"
            | "strictfp"
            | "super"
            | "switch"
            | "synchronized"
            | "this"
            | "throw"
            | "throws"
            | "transient"
            | "try"
            | "void"
            | "volatile"
            | "while"
            | "true"
            | "false"
            | "null"
            // `_` is a reserved identifier since Java 9 (Alipay's
            // instant-run fields are named `_`) — the `_<name>` mapping
            // turns it into `__`, matching jdc-core's call sites.
            | "_"
    );
    // A simple name may not START with a digit either (WhatsApp nests
    // `X/0Xx`): the declaration site and every reference (jdc-core's
    // sanitize_source_name) prefix the same underscore.
    let digit_start = name.chars().next().is_some_and(|c| c.is_ascii_digit());
    if clean && !keyword && !digit_start {
        std::borrow::Cow::Borrowed(name)
    } else if keyword || digit_start {
        std::borrow::Cow::Owned(format!("_{name}"))
    } else {
        // Non-ASCII single chars (Alipay names a field `支`) map to a
        // lone `_` — itself reserved since Java 9. Escape it.
        let mapped: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        if mapped == "_" {
            std::borrow::Cow::Owned("__".to_string())
        } else {
            std::borrow::Cow::Owned(mapped)
        }
    }
}

fn split_name(internal: &str) -> (String, String) {
    match internal.rfind('/') {
        Some(i) => (internal[..i].to_string(), internal[i + 1..].to_string()),
        None => (String::new(), internal.to_string()),
    }
}

/// Dotted source form of an internal name.
pub fn dotted(internal: &str) -> String {
    let mut out = internal.replace('/', ".");
    // `$` → `.` only when the following segment can start a Java
    // identifier (anonymous/synthetic tails stay `$`).
    let mut i = 0;
    while let Some(p) = out[i..].find('$') {
        let at = i + p;
        // A LEADING `$` (ProGuard keeps `$Gson$Types`) is part of the
        // source name — dotting it produced a leading `.Gson.Types`.
        let head_ok = at > 0
            && out[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
        let tail_ok = out[at + 1..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
        if tail_ok && head_ok {
            out.replace_range(at..at + 1, ".");
            i = at + 1;
        } else {
            i = at + 1;
        }
    }
    // Every `.`-segment must start an identifier: WhatsApp's `X/0Hl`
    // reached `extends` with the digit start unmapped.
    sanitize_fq(&out)
}

fn join_dotted(pool: &DexPool, names: &[String]) -> String {
    names
        .iter()
        .map(|n| print_class_name(pool, n))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A printable type name (arrays render with `[]` suffixes). Nested class
/// names dot their `$` when the outer chain is known to the pool.
pub fn type_name(pool: &DexPool, t: &JavaType) -> String {
    match t {
        JavaType::Void => "void".into(),
        JavaType::Object(n) => print_class_name(pool, n),
        JavaType::Array(inner) => format!("{}[]", type_name(pool, inner)),
        other => other.to_java(false),
    }
}

/// `com/foo/Outer$Inner` → `com.foo.Outer.Inner` — each `$` dots only when
/// its left side names a known class (literal `$` top-level names survive).
// ---------------------------------------------------------------------------
// The per-class import map (JLS 6.4.2 obscuring repair).
//
// A referenced FQN whose FIRST segment equals an in-scope class simple
// name binds the qualifier to the CLASS, not the package — `class j3
// implements j3.g` is cyclic inheritance and collapses javac's
// attribution for the whole package (the Object-cascade root, weixin
// 84 files + contagion). The repair: emit `import j3.g;` and render
// the SIMPLE name in every type position. The set is populated per
// class from a metadata pre-scan (supertypes, field types, method
// signature descriptors) and read by print_class_name and the
// DexCtx::obscured_simple implementation (which jdc-core's shorten /
// java_type_name consult).
struct ObscureState {
    /// The ROOT class being rendered (its simple name is the obscuring
    /// in-scope name).
    class: String,
    /// Metadata pre-scan results: internal → simple render.
    map: jdc_core::FxHashMap<String, String>,
    /// Expression-level obscured refs DISCOVERED during the body render
    /// (their imports emit at assembly time).
    recorded: jdc_core::FxHashSet<String>,
    /// Simple names of the CURRENT PACKAGE's own classes: importing a
    /// name that collides would SHADOW every same-package use of it in
    /// this file (JLS 7.5.1) — those refs stay qualified (obscured,
    /// erroring) rather than corrupt.
    blocked: jdc_core::FxHashSet<String>,
}

thread_local! {
    static OBSCURE: std::cell::RefCell<Option<ObscureState>> =
        const { std::cell::RefCell::new(None) };
}

/// Install the per-class render state (call at class render entry).
pub(crate) fn set_obscured_state(
    class: String,
    map: jdc_core::FxHashMap<String, String>,
    blocked: jdc_core::FxHashSet<String>,
) {
    OBSCURE.with(|m| {
        *m.borrow_mut() = Some(ObscureState {
            class,
            map,
            recorded: jdc_core::FxHashSet::default(),
            blocked,
        });
    });
}

/// Take the recorded expression-level refs and clear the state.
pub(crate) fn take_recorded_and_clear() -> jdc_core::FxHashSet<String> {
    OBSCURE.with(|m| {
        m.borrow_mut()
            .take()
            .map(|st| st.recorded)
            .unwrap_or_default()
    })
}

/// Render-time lookup: the metadata map first; otherwise an
/// expression-level ref whose first segment equals the root class's
/// simple name — record it (its import emits at assembly) and render
/// the simple name NOW.
pub(crate) fn obscured_render_pub(internal: &str) -> Option<String> {
    OBSCURE.with(|m| {
        let mut st = m.borrow_mut();
        let st = st.as_mut()?;
        if let Some(simple) = st.map.get(internal) {
            return Some(simple.clone());
        }
        let first = internal.split('/').next().unwrap_or("");
        let own = st.class.rsplit('/').next().unwrap_or("");
        if first == own && internal.split('/').count() >= 2 {
            // A NESTED internal's in-scope simple name is its `$` tail
            // (`x/a$b` imports/renders as `b`) — the slash-tail left the
            // `$` in the render (`t2$a` flat against a nested emission).
            let simple = internal
                .rsplit(['/', '$'])
                .next()
                .unwrap_or("")
                .to_string();
            if !simple.is_empty() && !st.blocked.contains(&simple) {
                st.recorded.insert(internal.to_string());
                return Some(simple);
            }
        }
        None
    })
}

/// The pre-scan: internal names referenced by the class's metadata
/// (supertypes, field types, method descriptors) whose first segment
/// equals the class's own simple name and which exist in the pool —
/// these get imports and simple-name renders.
fn compute_obscured_renders(pool: &DexPool, class: &PoolClass) -> jdc_core::FxHashMap<String, String> {
    let own_simple = class.name.rsplit('/').next().unwrap_or("");
    if own_simple.is_empty() {
        return jdc_core::FxHashMap::default();
    }
    let mut refs: jdc_core::FxHashSet<String> = jdc_core::FxHashSet::default();
    for sup in class.interfaces.iter() {
        refs.insert(sup.clone());
    }
    if let Some(sup) = &class.super_name {
        refs.insert(sup.clone());
    }
    for f in class.static_fields.iter().chain(class.instance_fields.iter()) {
        if let JavaType::Object(n) = crate::desc_type(&f.desc) {
            refs.insert(n.to_string());
        }
    }
    for m in class.all_methods() {
        if let Some(d) = m.parsed_desc() {
            for a in d.args.iter() {
                if let JavaType::Object(n) = a {
                    refs.insert(n.to_string());
                }
            }
            if let JavaType::Object(n) = &d.ret {
                refs.insert(n.to_string());
            }
        }
    }
    let mut out: jdc_core::FxHashMap<String, String> = jdc_core::FxHashMap::default();
    for r in refs {
        // First segment == the class's own simple name: the qualified
        // render would be obscured. The name must exist in the pool for
        // the import to resolve. The KEY stays internal (renders look
        // refs up by internal name), but the import line and the
        // simple name use the RENAMED display — a collision-renamed
        // class renders under its display name and an import of the
        // internal name does not resolve.
        let first = r.split('/').next().unwrap_or("");
        if first == own_simple && r.split('/').count() >= 2 && pool.get(&r).is_some() {
            // Own-family refs (the class itself, its nested members)
            // render through member scope — an import would be redundant
            // (self-import) or shadow a same-package sibling named like
            // the tail (`import t.t2.a` hijacks every bare `a` that
            // meant sibling class t.a — lark t/t2).
            if r == class.name || r.starts_with(&format!("{}$", class.name)) {
                continue;
            }
            let display = crate::apply_class_rename(&r);
            let simple = display
                .rsplit(['/', '$'])
                .next()
                .unwrap_or("")
                .to_string();
            if !simple.is_empty() {
                out.insert(r.clone(), simple);
            }
        }
    }
    out
}

pub fn print_class_name(pool: &DexPool, internal: &str) -> String {
    if let Some(simple) = obscured_render_pub(internal) {
        return sanitize_ref(&simple);
    }
    let cow = crate::apply_class_rename(internal);
    let internal: &str = &cow;
    // Flat EMISSION UNITS: a digit-tail member (anonymous / d8-lambda
    // shape, `Outer$lruCache$1`) is emitted as its own top-level file
    // whose simple name keeps every `$` — the `$` boundaries inside
    // that unit are part of the NAME, not nesting. References must use
    // the flat unit name; the generic loop below dotted the first
    // boundary (`LruCacheKt.lruCache$1`) against the declaration
    // `class LruCacheKt$lruCache$1` — every use of the type failed and
    // javac's attribution for the whole file collapsed (3627 pure-
    // cascade files on weibo).
    if internal.contains('$') && pool.get(internal).is_some() {
        if let Some(root) = emission_root(pool, internal) {
            if root.contains('$') {
                let below = &internal[root.len()..];
                if below.is_empty() {
                    // The unit itself: the flat name IS the reference.
                    return sanitize_ref(&internal.replace('/', "."));
                }
                let segs: Vec<&str> = below[1..].split('$').collect();
                if segs.iter().any(|s| s.is_empty()) {
                    // R8 tails like `ThreadMsg$$$` cannot be dotted at
                    // all — keep the whole name flat.
                    return sanitize_ref(&internal.replace('/', "."));
                }
                let mut out = root.replace('/', ".");
                for s in segs {
                    out.push('.');
                    out.push_str(s);
                }
                return sanitize_ref(&out);
            }
        }
    }
    let mut out = String::new();
    // `known` must test the ACCUMULATED internal prefix, not the bare
    // inter-`$` segment: the per-segment shape checked `pool.get("a")`
    // for the second level of `s5/o$a$b`, missed, and rendered the
    // undeclarable reference `s5.o.a$b` (找不到符号) for every nested-
    // nested type.
    let mut off = 0usize;
    loop {
        let rest = &internal[off..];
        match rest.find('$') {
            Some(i) => {
                let seg = &rest[..i];
                let prefix = &internal[..off + i];
                let known = pool.get(prefix).is_some()
                    || jdc_core::rename::is_renamed_display(prefix)
                    // The FULL name is not a pool class: this `$` cannot
                    // be a literal name (pool literal classes — an app's
                    // own `View$OnUnhandledKeyEventListener` — keep their
                    // `$` here AND at their declaration), so it can only
                    // be an external framework nesting boundary
                    // (`View$OnClickListener` → `.OnClickListener`).
                    || !pool.get(internal).is_some();
                // The `$` may only become a nesting dot when the tail
                // segment STARTS a Java identifier: R8's desugared-
                // library names carry `$` inside PACKAGE paths
                // (`j$/util/...` dotted into `j..util`) and suffixes
                // like `Collection$-EL` or anonymous `RequestId$1`
                // cannot be dotted under any reading.
                let tail_ok = rest[i + 1..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
                out.push_str(&seg.replace('/', "."));
                out.push_str(if known && tail_ok { "." } else { "$" });
                off += i + 1;
            }
            None => {
                out.push_str(&rest.replace('/', "."));
                return sanitize_ref(&out);
            }
        }
    }
}

/// The top-level EMISSION UNIT ancestor of a pool class: walking the
/// `$`-chain upward (outer_of), the first ancestor that is itself
/// emitted as its own compilation unit — a digit-tail rest marks a
/// flat boundary (see top_level_classes). Clean members render inline
/// in their outer, so only the unit's own name may carry `$`.
fn emission_root(pool: &DexPool, internal: &str) -> Option<String> {
    let mut cur = internal.to_string();
    loop {
        let outer = pool.outer_of(&cur)?;
        let rest = cur
            .strip_prefix(outer)
            .and_then(|t| t.strip_prefix('$'))
            .unwrap_or("");
        if !crate::clean_member_tail(rest) {
            return Some(cur);
        }
        cur = outer.to_string();
    }
}

/// Class-file names may contain characters Java source identifiers
/// cannot (`Collection$-EL`); the deterministic mapping matches the
/// declaration sites (java_ident).
/// Every `.`-segment of a fully-qualified name must start a Java
/// identifier: obfuscators emit `package do;` and `..badge.new..` paths.
///
/// PUBLIC on purpose: this is the single source of truth for the
/// declaration↔file-name mapping. The CLI writer used to keep its own
/// copy (`sanitize_file_seg`) which had drifted — it lacked the lone-`_`
/// escape below — so `l.֡` declared `class __` inside a file named
/// `_.java`. Callers reach it through `sanitize_seg`/`sanitize_internal`.
///
/// INJECTIVE on the characters, which is the property the writer needs.
/// The old mapping folded every non-identifier character to `_`, so
/// `l.᩻ܶ`, `l.᩻ۡ` and `l.֫᩷` all became `l.__`: on bin.mt.plus 22,636
/// distinct classes in package `l` collapsed onto three file names, the
/// writer's EEXIST branch overwrote them, and 22,633 sources were
/// destroyed with exit code 0. `_u<hex>` keeps them distinct, is
/// self-describing (it encodes the original code point) and costs the
/// decompiler nothing — the alternative, minting 22,636 registry
/// renames, makes every type reference allocate and took this sample
/// from 5.2 s to 26.8 s.
///
/// The only residual collisions are *lookalikes*: a literal class named
/// `_u1a7b` beside the class `᩻` (same code point), or the pre-existing
/// keyword form `_do` beside `do`. Those are detected and repaired
/// deterministically by `lossy_sanitize_renames`, which is why that pass
/// still exists.
pub fn sanitize_fq(dotted: &str) -> String {
    dotted
        .split('.')
        .map(sanitize_fq_seg)
        .collect::<Vec<_>>()
        .join(".")
}

/// One `.`-segment: escape, then repair identifier-position problems.
fn sanitize_fq_seg(seg: &str) -> String {
    // Ident-safe ASCII passes through unchanged (the overwhelmingly
    // common case, and the only one on the borrow-fast path); every
    // other character — non-ASCII, and ASCII punctuation such as the
    // `-` in `Collection$-EL` — becomes `_u<hex>`.
    let plain = seg
        .chars()
        .all(|c| c.is_ascii() && (c.is_ascii_alphanumeric() || c == '_' || c == '$'));
    let mut out = if plain {
        seg.to_string()
    } else {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(seg.len() + 8);
        for c in seg.chars() {
            if c.is_ascii() && (c.is_ascii_alphanumeric() || c == '_' || c == '$') {
                s.push(c);
            } else {
                s.push_str("_u");
                let _ = write!(s, "{:x}", c as u32);
            }
        }
        s
    };
    // `out` is pure ASCII from here: the identifier-position repairs.
    if is_java_keyword_name(&out)
        || is_restricted_type_name(&out)
        || out.chars().next().is_some_and(|c| c.is_ascii_digit())
    {
        out.insert(0, '_');
    }
    // A lone `_` is a reserved IDENTIFIER since Java 9.
    if out == "_" {
        out = "__".to_string();
    }
    out
}

/// Restricted contextual TYPE names — legal as member/local names
/// (rt.jar compiles `var` locals), illegal in class declarations and
/// type references. Consulted only on CLASS-name paths.
pub(crate) fn is_restricted_type_name(s: &str) -> bool {
    matches!(s, "var" | "yield" | "record" | "sealed" | "permits")
}

fn is_java_keyword_name(s: &str) -> bool {
    matches!(
        s,
        "abstract"
            | "assert"
            | "boolean"
            | "break"
            | "byte"
            | "case"
            | "catch"
            | "char"
            | "class"
            | "const"
            | "continue"
            | "default"
            | "do"
            | "double"
            | "else"
            | "enum"
            | "extends"
            | "final"
            | "finally"
            | "float"
            | "for"
            | "goto"
            | "if"
            | "implements"
            | "import"
            | "instanceof"
            | "int"
            | "interface"
            | "long"
            | "native"
            | "new"
            | "package"
            | "private"
            | "protected"
            | "public"
            | "return"
            | "short"
            | "static"
            | "strictfp"
            | "super"
            | "switch"
            | "synchronized"
            | "this"
            | "throw"
            | "throws"
            | "transient"
            | "try"
            | "void"
            | "volatile"
            | "while"
            | "true"
            | "false"
            | "null"
            // `_` is a reserved identifier since Java 9 (Alipay's
            // instant-run fields are named `_`) — the `_<name>` mapping
            // turns it into `__`, matching jdc-core's call sites.
            | "_"
    )
}

fn sanitize_ref(name: &str) -> String {
    sanitize_fq(name)
}

/// Dotted source form of an internal name.
pub fn dotted_pool(pool: &DexPool, internal: &str) -> String {
    print_class_name(pool, internal)
}

/// Unused import silencer.
#[allow(dead_code)]
fn _unused(_: &dyn Fn(&JavaType) -> jdc_core::types::GenericType) {
    let _ = java_type_to_generic;
}

#[cfg(test)]
mod sanitize_tests {
    use super::sanitize_fq;

    /// The property the file writer depends on: distinct class names must
    /// reach distinct output paths. The lossy predecessor of this mapping
    /// folded every non-identifier character to `_`, so 22,636 classes of
    /// `bin.mt.plus` (names built from Thai/Yi/Syriac code points) became
    /// three (`l.__`, `l.___`, `l._`) and all but three were overwritten.
    #[test]
    fn distinct_obfuscated_names_stay_distinct() {
        // The three names that collapsed onto `l/__.java` in the field.
        let names = ["l.᩻ܶ", "l.᩻ۡ", "l.֫᩷", "l.֡", "l.᩻᩶ۛ"];
        let mut mapped: Vec<String> = names.iter().map(|n| sanitize_fq(n)).collect();
        let before = mapped.len();
        mapped.sort();
        mapped.dedup();
        assert_eq!(mapped.len(), before, "sanitizer folded distinct names: {mapped:?}");
        assert!(
            mapped.iter().all(|m| m.is_ascii()),
            "sanitizer must stay ASCII: {mapped:?}"
        );
        // Self-describing: the escape carries the original code point.
        assert_eq!(sanitize_fq("l.᩻ܶ"), "l._u1a7b_u736");
    }

    /// Every output must be a legal Java identifier segment: no leading
    /// digit, no keyword, not the lone `_` reserved since Java 9.
    #[test]
    fn output_is_a_legal_identifier() {
        for (input, want) in [
            ("do", "_do"),
            ("_", "__"),
            ("0Xx", "_0Xx"),
            ("Collection$-EL", "Collection$_u2dEL"),
            ("᩻", "_u1a7b"),
            ("a", "a"),
            ("A$B", "A$B"),
            ("x_y", "x_y"),
        ] {
            assert_eq!(sanitize_fq(input), want, "input {input:?}");
        }
    }

    /// Documented residual: ident-safe ASCII passes through unchanged, so a
    /// literal `_u1a7b` collides with the single character U+1A7B. The
    /// collision is real — which is why `lossy_sanitize_renames` still owns
    /// a repair pass — and this test pins the shape the guard must catch.
    #[test]
    fn lookalike_collision_is_known_and_bounded() {
        assert_eq!(sanitize_fq("_u1a7b"), sanitize_fq("᩻"));
    }
}
