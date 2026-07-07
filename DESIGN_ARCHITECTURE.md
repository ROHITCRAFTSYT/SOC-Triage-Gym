# Design Architecture — Lusion-Style Immersive Scroll Site

**Subject analyzed:** [lusion.co](https://lusion.co) — Lusion Ltd, Bristol UK. Awwwards Site of the Month (May) and a reference point for real-time WebGL web experiences.
**Applied to:** `site/index.html` — the SOC-Triage-Gym presentation site.
**Rule followed:** recreate the *interaction architecture and techniques*, never the content, assets, or visual identity. Every line of code, every visual, and every word on our site is original to this project.

---

## 1. What lusion.co actually is (analysis)

Sources: the live site, Lusion's own Awwwards case study, and their open-source
[WebGL-Scroll-Sync](https://github.com/lusionltd/WebGL-Scroll-Sync) demo repository.

### 1.1 Macro-structure

| Layer | What they do |
|---|---|
| **Experience shell** | A single long-scrolling one-page site (not full-screen slide-jacking). Content is ordinary DOM; the "wow" layer is behind/around it. |
| **Persistent render surface** | One WebGL `<canvas>` (Three.js) fixed behind the DOM for the whole session. It never unmounts; *scenes* inside it change as you scroll. |
| **Scene system** | Each scroll region maps to a scene state. Transitions are *morphs* (particles/vertices interpolate between shape targets), not cuts. Vertex animations are baked offline in Houdini FX and streamed as compressed textures (16-bit integer quantization instead of 32-bit floats — their case study cites ~983 KB desktop / ~246 KB mobile for a cloth sim). |
| **Media** | Showreel and project previews are muted, looping, autoplaying videos — often used as WebGL textures so they can be distorted/transitioned in-shader. |
| **Narrative order** | Value proposition → proof of work (projects) → process → contact CTA. The scroll *is* the pitch. |

### 1.2 The scroll-sync problem (their core engineering insight)

Native scrolling runs on the browser's compositor thread; WebGL rendering runs
on the main thread inside `requestAnimationFrame`. Reading `scrollY` inside rAF
therefore lags the visually-scrolled position by a frame or more, which makes
DOM content and canvas content visibly "swim" against each other.

Lusion's fix — and the reason they accept scroll-hijacking — is to **own the
scroll number**:

1. Page height is created by an empty spacer; the real content sits in a
   `position: fixed` wrapper.
2. Every rAF tick, the target scroll (`window.scrollY`) is eased toward a
   smoothed value: `current += (target - current) × k` (k ≈ 0.08–0.12).
3. The smoothed value drives **both** the DOM (`transform: translate3d(0, -current, 0)`)
   and the WebGL camera/scene in the *same tick* — perfect sync, plus free
   "lag & settle" easing.
4. Scroll velocity (`current - previous`) is a bonus signal: they feed it into
   shader distortion / mesh skew so the page physically reacts to how hard you scroll.

### 1.3 Micro-interaction inventory

- **Preloader**: percentage counter + progress bar; the site only reveals when
  assets are genuinely ready (fonts, GPU pipelines warm).
- **Custom cursor**: a small dot + trailing ring, enlarged over interactive
  elements, with contextual labels (e.g. drag affordances); blend-mode inversion
  keeps it visible on any background.
- **Text choreography**: headlines enter as per-character/per-word staggered
  reveals tied to load or scroll-intersection.
- **Magnetic / reactive hover** on buttons and cards.
- **Grain/noise overlay** to unify DOM and WebGL layers and hide banding.
- **Reduced-motion & touch fallbacks**: the experience degrades to native
  scroll and static content rather than breaking.

---

## 2. Our recreation — mapping table

Every Lusion pattern was re-implemented **from scratch, dependency-free**
(vanilla JS + Canvas 2D with a hand-rolled 3D projection instead of
Three.js/WebGL) so the site stays a single offline-safe HTML file for a live
stage demo.

| lusion.co pattern | Our implementation (`site/index.html`) |
|---|---|
| Persistent WebGL canvas behind DOM | `#stage` — one fixed `<canvas>`, z-index 0, alive for the whole session |
| Three.js particle/vertex scene morphs | 850-particle system with a hand-written 3D pipeline: rotation matrices + perspective divide (`scale = fov/(fov+z)`) projected onto Canvas 2D |
| Houdini-baked shape targets | Procedural shape generators, one per scene: Fibonacci **sphere** (threat globe) → **wave field** (alert ocean) → **cubic lattice** with edge lines (network) → **three interlocked rings** (Tier-1/Tier-2/Manager) → **helix tunnel** (250-step campaign) → **ascent columns** (learning curve) |
| Scene transitions on scroll | Sections carry `data-scene="0…5"`; scroll position maps to (current, next, blend) and every particle lerps between shape targets — a continuous morph, never a cut |
| Scroll-sync architecture | Identical scheme: `#spacer` provides page height, `#smooth` is fixed and translated by a lerped value inside the same rAF that draws the canvas (`SMOOTHING = 0.085`) |
| Scroll-velocity reactivity | `velY` expands particle spread (`spread = 1 + min(|velY|·0.006, 0.55)`) — fast scrolling visibly scatters the point cloud, which settles as you stop |
| Camera drift + mouse parallax | Rotation = `time·0.06 + scroll·0.00035 + (mouse−0.5)·0.5` |
| Showreel videos | Two live procedural feeds rendered on canvas — an episode **log stream** and a **threat radar sweep** (conic-gradient sweep + blips) — plus a real `<video>` slot (`assets/showreel.mp4`, autoplay/muted/loop) that self-removes and falls back to the canvas feed if the file is absent |
| Project gallery | Drag-to-scroll horizontal gallery (`pointerdown/move` + `scrollLeft`) of **real committed artifacts**: GRPO loss curve, baseline-gap chart, red-team curriculum oscillation, task landscape, theme-coverage matrix |
| Preloader with counter | `#loader`: 0→100 counter + hairline bar; holds at 92% until `document.fonts.ready` / `load` fires, then releases and slides up with `cubic-bezier(.76,0,.24,1)` |
| Custom cursor | Dot + trailing ring (lerped at 0.2), `mix-blend-mode: difference`, grows on links, shows a **DRAG** tag over the gallery; hidden entirely on coarse pointers |
| Staggered text choreography | Per-letter hero reveal (`--i`-indexed animation delays, blur + rise + rotate) and per-word statement reveal on intersection |
| Grain overlay | `#grain` fixed scanline/noise layer above canvas + DOM, below cursor |
| Accessibility fallbacks | `prefers-reduced-motion` and touch devices bypass the fixed-wrapper smoothing (native scroll), animations collapse to near-zero durations |
| *(our addition)* rAF watchdog | Browsers suspend rAF in hidden tabs; a 500 ms interval detects a stalled loop and applies scroll transform + one static canvas frame so captures/background tabs never show a frozen page |

## 3. Scene map

| Scroll region | Scene | Particle formation | Palette |
|---|---|---|---|
| Hero | 0 | Threat globe (Fibonacci sphere, pulsing red "alert" points) | green |
| Brief + statement | 1 | Wave field (sine-displaced plane — alert ocean) | slate |
| Architecture | 2 | Network lattice with drawn edges | cyan |
| Team mode / Live wire | 3 | Three interlocked rings — the three roles | amber |
| Tasks + Reward | 4 | Helix tunnel — the 250-step campaign | red |
| Training → Demo | 5 | Ascent columns — the learning curve | green |
| Outro | 0 | Globe returns — full circle | green |

Transition window: the last 28% of each region blends into the next scene
(`local > 0.72 → f = (local−0.72)/0.28`), so morphs happen *between* reading
moments, not during them.

## 4. Why Canvas 2D instead of Three.js

Lusion ships WebGL because their business is GPU craft and they control their
hosting. This site's constraint set is different — it must:

1. survive venue Wi-Fi (no CDN imports),
2. stay a single reviewable file in the repo,
3. run on any projector laptop without GPU-driver surprises,
4. degrade gracefully during a live talk.

A 3D-projected particle field on Canvas 2D at ≤850 points hits 60 fps on
integrated graphics and needs zero dependencies. The *architecture* (persistent
stage, scene morphs, rAF-synced lerped scroll, velocity reactivity) is exactly
Lusion's; only the rasterizer is humbler.

## 5. File map

```
site/
  index.html      the immersive scroll experience (this document's subject)
  classic.html    previous flat version, kept as fallback for the talk
  assets/         committed chart artifacts shown in the drag gallery
                  (+ optional showreel.mp4 slot for a real video)
```

## 6. Sources

- https://lusion.co
- Lusion × Awwwards — "Case Study for Lusion" (Site of the Month, May)
- https://github.com/lusionltd/WebGL-Scroll-Sync — their scroll-sync reference implementation
- Codrops — "Curly Tubes from the Lusion Website with Three.js"
