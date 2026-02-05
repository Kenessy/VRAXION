---
imported_from: "S:/AGENT A FOLDER FOR INCOMING DATA/deep-research-report (5).md"
imported_utc: "2026-02-05T05:55:26Z"
source_sha256: "CEC62AC1C40AD59D33B3C2B429B5C2D5D13AA0D569489B79562F1A4950609956"
repo_commit: "b91f5c4792a45e758e2b832db4203afa8ab1a6a3"
notes: "Verbatim import; citation markers may be non-portable."
---

# Scalable Communication and Refinement Protocols for a Growing Swarm of Node-Networks

## Findings from the VRAXION codebase and internal artifacts

A large fraction of the “SWARM” problem—stable growth, specialization, and coordination—already exists in VRAXION as *modular expert machinery* plus *address-based routing*.

**A ring-addressed routing substrate is already implemented.** The `AbsoluteHallway` core describes a “boundaryless ring memory” whose pointer update is an explicit mixture of *jump* vs *walk/stay*, with optional inertia and deadzone, plus optional auxiliary rings (sensory/vault/think). fileciteturn68file0L12-L20 fileciteturn68file0L319-L329  
Crucially for swarm scaling, it defines a `router_map` buffer that **decouples “address → expert” from simple modulo routing**: the map is initialized as `arange(ring_range) % num_experts`, and then used to translate pointer bins to expert IDs. fileciteturn68file0L586-L593 fileciteturn68file0L718-L729

**There is already telemetry for specialization and “participation.”** `AbsoluteHallway` computes expert usage counts, normalized entropy, active-expert count, and max-share (dominance) from routed expert IDs. fileciteturn68file0L731-L752  
That same file defines a normalized per-step entropy proxy (preferentially over expert IDs) intended for a higher-level controller. fileciteturn68file0L753-L775

**Expert modularity is first-class and supports growth/shrink operations offline.** VRAXION provides “mitosis” and “prune/merge” tools that operate **only on checkpoints**:

- **Mitosis split** clones a parent expert’s tensors into a new highest-ID expert and redirects selected `router_map` addresses to the new expert. fileciteturn71file0L4-L13 fileciteturn71file0L124-L137  
- **Prune/merge** removes only the **highest-index** expert (to keep IDs dense), remapping its router-map entries into a kept expert. fileciteturn72file0L4-L15 fileciteturn72file0L147-L161  

This “highest-id-only” rule is an explicit invariance mechanism: it prevents renumbering cascades that would scramble learned roles. fileciteturn72file0L13-L15

**There is already a notion of expert lifecycle metadata.** The `modularize_checkpoint.py` tool explodes a checkpoint into `system/router.state` plus per-expert tensors and a `meta.json` listing `created_step`, `last_used_step`, and `contrib` per expert. fileciteturn70file0L4-L11 fileciteturn70file0L144-L152  
This is directly usable as an operational definition of *participation over time* (even before adding any new routing scheme).

**Routing is currently “top-1 per sample” in the expert head, with optional out-of-core offload.** The extracted `LocationExpertRouter` routes each batch element to a single expert by `pointer % num_experts`; when hibernation is enabled, expert weights can be restored from disk during forward passes. fileciteturn69file0L131-L153  
Note a scalability caveat: for legacy compatibility, it attempts restoration for *every* expert in index order regardless of whether the batch routes to it. fileciteturn69file0L150-L153

**There is already a “fast vs slow” control mechanism and it explicitly uses φ thresholds.** VRAXION’s `BrainstemMixer` is a Schmitt-trigger controller that emits a shield weight `w` (fast vs slow mode) and uses thresholds based on **1/φ ≈ 0.618** and **1/φ² ≈ 0.382**. fileciteturn67file0L30-L40  
In `AbsoluteHallway`, a “think ring” can run in a dual-core mode (fast and slow traces) whose mixing ratio can be driven by that brainstem controller using the entropy proxy derived from routing. fileciteturn68file0L514-L542 fileciteturn68file0L1197-L1217

**Your own internal “gate” documents already define “learnability + stability” gates for candidates.** The “Topological Expedition” consensus note defines a sprint-gate evaluation with `seq_len=32`, a minimum `eval_acc >= 8%`, and two stability criteria: a “Gradient Gate” requiring `ptr_w` not to saturate ( <5% of steps pinned at 1.0 ) and a “Symmetry Gate” rejecting constant/zombie behavior in `ptr_align_cos`. fileciteturn75file0L13-L23

**SWARM is explicitly a planned phase in the VRAXION roadmap.** The roadmap lists “SWARM” as a planned milestone with “multi-agent protocol,” “distributed reasoning,” and “divergent logic forks.” fileciteturn74file0L149-L160

This matters for the research question because VRAXION already contains (a) an address space, (b) modular nodes (“experts”), (c) a stable remapping primitive (`router_map` + offline mitosis/prune), and (d) controllers that gate fast/slow behavior using a normalized entropy signal.

## Invariance under growth

A scalable swarm protocol must specify what *cannot* change when N grows. The core invariance target is: **adding nodes must not rewrite the meaning of old routing keys**, or you get role reshuffling and catastrophic forgetting.

### Stable identity and stable keyspaces

VRAXION’s checkpoint tools hint at the right decomposition:

- **Stable node identity**: experts are indexed, and pruning only removes the highest index to keep identifiers dense and stable. fileciteturn72file0L13-L15  
- **Stable routing keyspace**: `router_map` is explicitly called a “decouple address→expert from modulo routing” mechanism, i.e., the keyspace is the ring address, and the mapping can evolve without forcing a global renumber. fileciteturn68file0L589-L593

However, for a truly *open-ended* swarm, you typically cannot store a full lookup table for an unbounded keyspace, and you cannot accept mappings that shift globally when N changes (e.g., `key % N`, or `key → key + floor(N/φ)`).

Distributed-systems literature has a very direct answer: **consistent hashing** and **rendezvous (HRW) hashing** are designed so that when a new node joins, only a small fraction of keys move, and the rest keep their assignments. The Chord DHT paper summarizes consistent hashing’s stability properties (minimal disruption when nodes join/leave) as a design principle for scalable routing. citeturn6search43  
Rendezvous hashing (highest-random-weight mapping) gives a deterministic “choose the best node for this key” rule that is similarly stable under membership changes. citeturn5search3

**Implication for SWARM invariance:** define “lane semantics” as a **keyspace** (addresses, topics, or task signatures), and define routing as a stable function `route(key, node_set)` that changes minimally when the node set changes. The existing `router_map` approach is the “explicit table” version of this; consistent/HRW hashing are functional versions suitable for growth.

### Permutation invariance vs stable semantics

If your global readout depends on the *ordering* of nodes (e.g., dense concatenation into a fixed vector whose indices carry meaning), growth will break semantics unless you freeze index assignments forever.

Set-function results are relevant here:

- Deep Sets proves that permutation-invariant set functions can be represented with a sum/aggregation form over element embeddings, which supports “node order doesn’t matter” designs for global summarization. citeturn12search8  
- Set Transformer shows an attention-based permutation-invariant approach for set representations, which is a richer alternative to sum pooling when interactions matter. citeturn2search9

**Operational takeaway:** if you want roles to be stable while N changes, your “global bus” should be *permutation-invariant*, and your “routing keys” should be *stable under membership expansion* (consistent/HRW hashing or an explicit remapping table that changes only locally).

### Stronger version of invariance: contract-first specs

VRAXION’s GPU workload system uses strict validation and a stable workload ID computed from canonical JSON + SHA-256. fileciteturn66file0L25-L29 fileciteturn66file0L100-L118  
That pattern transfers cleanly to SWARM: define a **routing/wiring spec** whose identity is stable and whose optional fields (labels/notes) do not affect semantics—then hash it to produce a “wiring_id.” This is the most practical way to prevent silent semantic drift as the swarm grows.

## Participation definition and quantifying “minimum influence”

Your explicit preference (“no Top‑k; most of the network gets excluded”) implies you want a participation definition stronger than “sometimes used.”

There are three progressively stricter participation notions; they are not equivalent, and each implies different routing/aggregation designs:

### Every node participates every step

This means each node emits a message every step and the global computation uses all of them (directly or via a deterministic compression). This resembles continuous communication schemes where each agent contributes to a shared message, like CommNet’s communication vector that aggregates across agents. citeturn2search1

**When viable:** when each node is small and message aggregation is cheap (or compressible).  
**Main risk:** collisions/interference if the aggregation compresses too hard.

### Every node participates over time

This is the mixture-of-experts style: each step uses a subset, but the training/routing system must guarantee that (1) nodes don’t starve forever and (2) responsibility distributes.

MoE work repeatedly identifies “expert collapse” and load imbalance as central failure modes; classic sparsely-gated MoE uses a load-balancing term to avoid routing everything to a few experts. citeturn0search0  
Switch Transformers uses top‑1 routing explicitly for efficiency and adds an auxiliary load-balancing objective to keep usage distributed. citeturn1search3

VRAXION already exposes the right observables: counts, entropy, max-share, and active-expert count. fileciteturn68file0L731-L752

### Every node has measurable causal influence

This is the strongest and most expensive notion: a node’s output must affect downstream computations in a measurable way (e.g., gradient signal, Shapley-like influence, or ablation impact). In practice, you approximate it with cheap proxies:

- **Usage-based influence**: duty cycle and entropy of assignments (`ptr_expert_entropy`, `ptr_expert_active`, `ptr_expert_max_share`). fileciteturn68file0L731-L752  
- **Contribution accumulators**: for each node, maintain `contrib` and `last_used_step`, which VRAXION’s modular checkpoint metadata already supports structurally. fileciteturn70file0L144-L152  
- **Stability of control signals**: your internal sprint gates already define “no saturation” criteria for pointer-related control (`ptr_w` saturation <5%) and reject “zombie/constant” symmetry metrics. fileciteturn75file0L20-L23

A practical “minimum influence” definition (without Top‑k) that matches your direction is:

> Over a horizon H, each node i must exceed a minimum contribution mass  
> \( \frac{1}{H}\sum_{t=1}^{H} \|g_i(t)\,m_i(t)\|_2 \ge \epsilon \)  
> and the distribution over nodes must maintain minimum entropy or maximum-share bounds.

This makes participation a measurable gate, not only a design intention.

## Bandwidth, compute, and VRAM budgets

Design choices that “include everyone” only make sense if they fit realistic hardware constraints. VRAXION already has a concrete VRAM accounting model and stability contract, so the swarm protocol should be grounded in those numbers.

### VRAM constraints from VRA‑34 and what dominates

The VRA‑34 VRAM breakdown identifies a dominant dynamic term:

\[
\text{ring\_buf\_bytes} = B \cdot \text{synth\_len} \cdot \text{ring\_len} \cdot \text{slot\_dim} \cdot \text{bytes\_per\_elem}
\]
fileciteturn64file0L76-L91

It then models peak allocated and reserved memory as approximately linear in that term (with fitted slope ~2.0) plus base allocations and allocator overhead. fileciteturn64file0L102-L116

On the measured system, the `small×real` tier hits ~12.3 GiB peak reserved at B=24 but fails the `vram_guard` at B=32 (~16.37 GiB). fileciteturn64file0L132-L137  
The `real×real` tier shows that B=3 is already ~13.7 GiB peak reserved. fileciteturn64file0L137-L141

The stability contract defines the guardrail explicitly:
`peak_vram_reserved_bytes > 0.92 × total_vram_bytes` is a hard fail gate. fileciteturn65file0L54-L66

### Reserved vs allocated and why “reserved” is the right guardrail

VRA‑34 distinguishes between allocated tensor memory and allocator-managed reserved memory. fileciteturn64file0L33-L49  
This matches entity["organization","PyTorch","deep learning framework"] documentation: `max_memory_reserved` tracks memory held by the caching allocator, while `max_memory_allocated` tracks memory occupied by tensors; reserved can remain high due to caching and fragmentation. citeturn17search2

**Implication for SWARM:** if a “communication buffer” is allocated once and reused, reserved memory may stay high even if per-step usage varies. Budgeting must be based on reserved, not just live tensors.

### Bandwidth scaling and why dense concatenation breaks first

If you literally concatenate all N node messages of dimension D into one global vector, the aggregator-visible bandwidth is \(N\cdot D\) floats per step. This is the most inclusion-friendly design, but its bandwidth scales linearly and becomes the first bottleneck as N grows.

The following chart is a conceptual scaling comparison (D=16, k=8 for top‑k, M=512 buckets for a sketch bus). It shows why dense concatenation is the only curve that grows without bound.

![Scaling of aggregator-visible bandwidth](sandbox:/mnt/data/scaling_aggregator_bandwidth.png)

**Interpretation:** if you want *everyone participates* while still allowing N→large, you must either:
1) compress everyone’s messages into a fixed-size global state (hash/sketch bus or hierarchical), or  
2) rely on locality + multiple passes (graph propagation), accepting more passes as coordination cost.

### Multi-pass costs and the “coordination diameter”

Local wiring can be cheap per pass but expensive in passes. Small-world style wiring (local + sparse long-range) is known to reduce characteristic path lengths dramatically while keeping clustering high. citeturn9search0  
The chart below illustrates the conceptual pass-count difference between a ring/local-only connectivity and small-world/expander-like connectivity.

![How wiring determines passes needed](sandbox:/mnt/data/passes_vs_wiring.png)

This is where “passes as cognition” meets systems reality: more passes are not free—each pass is additional compute, additional activation memory (unless you use implicit methods), and potentially additional coordination latency.

## Communication architectures for scalable inclusion

This section compares the requested set: dense concatenation, soft gating, top‑k sparse routing, hierarchical grouping, and hash‑bus/sketching. The comparison focuses on (a) scalability in N, (b) inclusion properties, and (c) stability/learning dynamics.

### Comparative table

| Architecture | How messages combine | Aggregator-visible bandwidth | Inclusion property | Stability / failure modes | When it’s a good fit |
|---|---|---:|---|---|---|
| Dense concatenation | `[m1;m2;…;mN]` | **O(N·D)** | Everyone participates every step | Becomes bandwidth/compute bottleneck; ordering semantics must be stable | Small N, or when you *need* full transparency |
| Soft gating over all nodes | `Σ g_i m_i` | O(D) (but computes all g_i) | Everyone can participate every step | Gate saturation / vanishing influence; “rich get richer” without regularization | Moderate N, strong regularizers and diagnostics |
| MoE top‑k routing | Select k nodes per token/step | **O(k·D)** | Participation over time only | Expert collapse/starvation without load balancing; hard exclusion per step | Very large N with strict compute limits citeturn0search0turn1search3 |
| Hierarchical grouping | Nodes → groups → supergroups | O(D) at the root (distributed traffic) | Everyone participates via local aggregation | Group boundary semantics can drift; hierarchy can “lock in” wrong partitions | Large N with structured domains; good for distributed systems thinking |
| Hash/sketch bus | Nodes write into M shared buckets | **O(M·D)** | Everyone participates (via superposition) | Collisions/interference; needs careful design + monitoring | Large N with inclusion constraint; resource-bounded global context citeturn3search1turn3search2 |

### Hash/sketch bus is the most direct “no Top‑k but still scalable” option

If you reject top‑k on principle, but still want N to grow, the hash/sketch family is uniquely relevant because it provides **global inclusion with fixed-size communication**:

- Count‑Min Sketch gives a formal way to summarize many updates into a small array with bounded error guarantees under certain assumptions. citeturn3search1  
- Feature hashing (“hashing trick”) shows how to project large sparse features (or identifiers) into a fixed-dimensional space while preserving useful learning signal at scale. citeturn3search2

A SWARM interpretation is straightforward: each node writes (possibly signed) contributions into one or more buckets indexed by stable hashes of its ID and/or current context key. Everyone writes; the bus stays fixed-size; collisions are the price.

### Dense/soft vs sparse/hierarchical: what changes in learning dynamics

Two important lessons from research apply when you move away from dense concatenation:

1) Sparse routing (MoE) is compute efficient but needs explicit balancing or else specialization collapses. citeturn0search0turn1search3  
2) Global set-style pooling avoids ordering issues but can wash out fine-grained structure unless you reintroduce interaction capacity (e.g., attention-style set models). citeturn12search8turn2search9

Your internal gates (“no saturation,” “no zombie symmetry”) are exactly the correct mindset: most failures present first as *collapsed control signals* long before the final task metric looks obviously broken. fileciteturn75file0L20-L23

## Dynamic wiring and φ-based skip connections

Your earlier brainstorming around φ is pointing at a real phenomenon: **irrational rotations distribute connections/points “evenly” without periodic locking**. But the growth-invariance question changes what is and isn’t viable.

### φ in VRAXION today: a controller threshold, not a topology

VRAXION uses φ-derived thresholds in a Schmitt-trigger controller that switches between fast (shield) and slow (deep) behavior; the engage threshold is 1/φ and release is 1/φ². fileciteturn67file0L30-L40  
That controller can drive fast/slow mixing in the dual think-ring loop via entropy computed from routing. fileciteturn68file0L1197-L1217

This is important: φ is being used as a *stability heuristic* (hysteresis thresholds), not as a “connect every φ-th node” rule.

### φ-based wiring can be useful, but only if it is ID-based (not N-based)

If you define skip edges using formulas that depend on **current N** (e.g., “connect i to floor(N/φ)”), then adding nodes rewires the entire graph and destroys stable semantics. That is exactly the kind of global remapping consistent hashing was invented to avoid. citeturn6search43

If you instead define:
- a permanent node ID, and  
- a deterministic mapping from that ID to a point on a circle via irrational rotation / golden-angle increments,  

then you can add nodes without moving old nodes. Golden-angle / golden-ratio sequences are used in low-discrepancy sampling precisely because they are simple and distribute points quasi-uniformly. citeturn16search7  
There is also empirical/biophysical literature showing why the golden angle appears as an optimal divergence angle in phyllotaxis patterns. citeturn15search1

### Small-world wiring is the right conceptual target

The Watts–Strogatz small-world result shows that injecting a small amount of long-range rewiring into a clustered lattice can produce networks with short path lengths (fast propagation) while keeping clustering high. citeturn9search0  

For SWARM, “φ-based skip” is best interpreted as a **deterministic way to place long-range edges** (so you avoid a random-number dependency) *while keeping degree constant*. If you need growth-stable semantics, combine it with ID-based mapping (or consistent/HRW hashing) rather than N-based distances.

### Visualizing growth-stable rewiring via router_map

One of the most concrete “growth without reshuffle” mechanisms is already in VRAXION: redirect only a subset of routing keys (addresses) to a new node. That’s exactly what the mitosis tool does. fileciteturn71file0L8-L13

The diagram below illustrates how a router-map partition can change locally during growth without rewriting the entire mapping:

![Router map partition before/after mitosis](sandbox:/mnt/data/router_map_growth_diagram.png)

Conceptually, this is “wiring stability via keyspace partition,” and it generalizes directly to swarm communication: new nodes claim a few keys and grow them, rather than forcing everyone to renormalize connections because N changed.

## Pass structures as operational refinement loops

You explicitly want passes that are **not gradient backprop**, but operational cycles (draft, correction, integration, governance). That framing maps cleanly onto established iterative/refinement literatures—without requiring the system to be trained like a standard unrolled deep network forever.

### Passes as iterative inference and controlled computation

Predictive coding models treat cognition as iterative reconciliation between top-down predictions and bottom-up errors. citeturn13search0  
Modern deep learning has parallel analogues:

- Universal Transformers add recurrence/iteration to transformer-style processing and can use dynamic halting to adapt computation per position. citeturn18search9  
- Deep Equilibrium Models compute a fixed point of an implicit layer and backpropagate via implicit differentiation, offering a way to do “many refinement steps” with memory efficiency. citeturn9search6  
- Adaptive Computation Time explicitly learns “how many internal steps to run” before emitting an output. citeturn18search1  

These all support the same core claim: **multiple refinement passes can improve reasoning/coordination, but only if computation is gated and stabilized**.

### Why “too many passes” breaks: oversmoothing and homogenization

Graph/message-passing systems exhibit “oversmoothing”: after enough propagation steps, node representations can become indistinguishable, losing discriminative power. citeturn18search4  
That is the technical version of your intuition about “signal decay” and “analysis paralysis”: beyond some depth of repeated mixing, the system can wash out distinctions and/or amplify self-generated artifacts.

### A four-pass loop that matches VRAXION’s existing controllers

VRAXION’s dual think-ring plus brainstem controller is already a form of “fast vs slow” pass mixing driven by an entropy/danger proxy. fileciteturn68file0L514-L542 fileciteturn68file0L1197-L1217  
The brainstem itself is explicitly a hysteretic switch with φ thresholds. fileciteturn67file0L30-L40  

A practical operational mapping is:

- **Pass 1 (Draft/Instinct):** local, fast, minimal coordination.
- **Pass 2 (Critique/Reason):** global check (bus, hierarchy, or long-range edges).
- **Pass 3 (Integration):** commit structural updates (router_map edits; mitosis/prune decisions; memory writeback).
- **Pass 4 (Governor):** meta-controller that decides whether to stay in fast mode, slow down, or trigger integration based on danger proxies (entropy, flip-rates, saturation).

This diagram makes the roles explicit:

![Operational multi-pass structure](sandbox:/mnt/data/pass_structure_diagram.png)

The key is that Pass 3 and Pass 4 are *not per-token expensive by default*. They can be triggered only when stability gates are satisfied—or when danger/instability is detected.

## Recommendations for VRAXION SWARM phase

The recommendations below aim to satisfy all ten research dimensions simultaneously: growth invariance, meaningful participation, budget realism, multi-pass refinement, and stability gates.

### Use “keyspace stability” as the primary invariance mechanism

1) **Keep a stable routing keyspace that does not depend on N.** In VRAXION terms, this is the “ring address” space. The existing `router_map` explicitly exists to decouple addresses from modulo routing. fileciteturn68file0L589-L593  
2) **Make growth events local in keyspace.** Adopt the mitosis model: clone a node (expert) and redirect only a targeted set of keys/addresses. fileciteturn71file0L8-L13  
3) **If you outgrow explicit tables, replace router_map with functional routing:** consistent hashing or rendezvous hashing so that adding nodes moves only a small fraction of keys. citeturn6search43turn5search3  

This directly prevents catastrophic “role reshuffling under growth.”

### Treat participation as a first-class metric gate

Use a two-layer definition:

- **Per-step inclusion** (design-level): all nodes write *something* each step (either directly or via a sketch bus).  
- **Over-time influence** (measurement-level): enforce minimum duty cycle / contribution over a horizon H.

Concrete metrics already supported by VRAXION structures:

- Enforce minimum normalized entropy and maximum dominance thresholds using `ptr_expert_entropy` and `ptr_expert_max_share`. fileciteturn68file0L731-L752  
- Track and gate on lifecycle metadata (`last_used_step`, `contrib`) in modular expert layouts. fileciteturn70file0L144-L152  

Adopt your internal gate style: treat participation failures as “stability FAIL,” similar in spirit to your sprint gates that reject saturation/zombie behavior early. fileciteturn75file0L20-L23

### Prefer a sketch-bus or hierarchical bus to avoid Top‑k exclusion

If Top‑k is unacceptable, the most scalable inclusion-compatible architecture is:

- **Hash/sketch bus of fixed size** (M buckets × D_bus), where each node writes `g_i m_i` into a small number of buckets.  
- Optionally, each node reads back a small number of buckets (content-addressed context) in Pass 2.

This is the direct analog of Count‑Min/feature-hashing compression, trading exactness for scalability. citeturn3search1turn3search2  

A practical starting point consistent with single-GPU experimentation:

- **D_node:** 8–16 (node message vector).  
- **D_bus:** 16 (global bus channel width).  
- **M:** 256–1024 (buckets).  

Your own “Topological Expedition” notes already show that simply going from 5D → 16D did not change results on a too-dark probe, and that the actionable path was to change the gate/probe regime (seq_len=32 sprint) rather than just increasing dimensionality. fileciteturn75file0L9-L18  
So treat D as a *budget-controlled knob*, but don’t expect it to substitute for better gating, routing stability, and pass structure.

### Choose pass counts that reflect “coordination diameter,” not ideology

Use this default:

- **Two always-on passes:** Draft (local/fast) + Critique (global check).  
- **One conditional integration pass:** commit router_map edits / expert lifecycle updates when gates are satisfied.  
- **One governor pass:** run continuously but cheap; it only triggers expensive actions (slow-downs, integration) when danger proxies trip.

This matches what VRAXION already implements: a fast/slow mixer driven by entropy proxies with φ hysteresis, and a dual-ring loop where mixing changes based on that controller. fileciteturn67file0L30-L40 fileciteturn68file0L1197-L1217  

Avoid pushing pass count upward without strong reason; oversmoothing literature gives a real failure mode for too much repeated mixing. citeturn18search4

### Ground N and D choices in VRAM and throughput gates

For single-GPU work (your measured environment), respect the existing VRAM guard and the fact that ring buffer state dominates VRAM:

- Use VRA‑34’s ring buffer model to choose safe batch sizes and sequence lengths. fileciteturn64file0L76-L91  
- Treat `peak_vram_reserved` as the guardrail metric in agreement with both VRA‑34 and entity["company","NVIDIA","gpu vendor"]-adjacent allocator behavior. fileciteturn64file0L33-L49 citeturn17search2  
- Keep runs inside the `0.92 × total_vram` contract boundary. fileciteturn65file0L64-L66  

A practical initial SWARM scale that is unlikely to collide with VRAM limits (because communication buffers are small compared to ring_buf_bytes) is:

- **N:** 64–512 nodes for first “behavioral” experiments (so you can still debug roles).  
- **D:** 16 as a control (matches your current bracket) with 7/11 as smaller candidates under the sprint-gate regime. fileciteturn75file0L28-L33  

If “nodes” become larger than the current expert heads (e.g., mini-MLPs), then N must be set by throughput, not VRAM alone—because launching thousands of tiny networks separately is GPU-inefficient unless you batch them like MoE implementations do. citeturn0search0turn1search3

### Preserve stable semantics while changing routing implementations

One last pragmatic recommendation: keep the golden verifier principle as you evolve SWARM. The existing verification script treats router semantics + offline tools as a contract to avoid accidental breakage. fileciteturn73file0L3-L13  

If you introduce a new router that avoids the O(N) “restore all experts every forward” behavior (necessary for very large N), make it a new, explicitly versioned router mode rather than silently changing semantics—so the system can retain a stable “legacy semantics” path for reproducibility. fileciteturn69file0L150-L153