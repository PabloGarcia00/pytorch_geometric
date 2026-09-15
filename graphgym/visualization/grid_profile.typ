// grid_profile.typ -- frontend half of the JSON -> Typst pipeline, grid mode.
// 2x4 small-multiples: one panel per model (7) plus one truth-only reference
// panel, in raw physical Watts (not peak-normalized -- this figure is about
// absolute magnitude, unlike models_profile.typ / physics_profile.typ).
// Each model panel shows true PV (thin dark line), predicted PV median
// (that model's fixed accent color), and the P10-P90 band (lq.fill-between,
// same color, low alpha) -- the project's established quantile-band
// convention. Net load is deliberately NOT plotted as its own line -- it's
// always <li= 0 relative to PV's own peak here and was only ever adding dead
// negative-axis space that shrank the PV detail everyone actually wants to
// compare. (An earlier version also marked export timesteps, true_net < 0,
// with a thin per-panel strip -- removed for visual clarity; the physics
// penalty PV >= Export this would have illustrated is discussed in prose.)
//
// Household/day/seed selection rules (household_rule/day_rule/seed_rule in
// the JSON) are not rendered in this figure -- kept in the JSON payload for
// the paper's own methods-section prose to cite, not duplicated on-figure,
// so the on-page text stays to just the household + date identifier.
//
// Usage: typst compile --input json=data/grid_dual_wx_off.json grid_profile.typ out.pdf
#import "@preview/lilaq:0.6.0" as lq

// Sized to fit a single A4 page and stay close to an IEEE double-column
// figure width (~18.2cm/7.16in) -- true single-column IEEE width (8.89cm)
// isn't realistic for 8 readable multi-line time-series panels side by
// side, so this targets the double-column-spanning-figure case instead.
// Margins kept minimal since the page is auto-cropped to content anyway.
#set page(width: auto, height: auto, margin: (x: 2mm, y: 2mm))
#set text(font: "New Computer Modern", size: 8pt)
#set par(leading: 0.4em)

#let data = json(sys.inputs.at("json", default: "data/grid_test.json"))

#let parse-dt(s) = {
  let parts = s.split("T")
  let date = parts.at(0).split("-")
  let time-raw = parts.at(1)
  let time-str = if time-raw.contains("+") {
    time-raw.split("+").at(0)
  } else {
    time-raw
  }
  let time = time-str.split(":")
  datetime(
    year: int(date.at(0)), month: int(date.at(1)), day: int(date.at(2)),
    hour: int(time.at(0)), minute: int(time.at(1)), second: int(float(time.at(2))),
  )
}

// Fixed model -> color, reusing models_profile.typ's exact 7-color palette
// but as a true identity mapping (every figure gives the same model the
// same color) rather than that file's alphabetical-position cycling.
#let MODEL_COLORS = (
  "ST-GNN": orange,
  "MLP": blue,
  "LSTM": green.darken(20%),
  "CVAE": purple,
  "KNN": red.lighten(20%),
  "Linear": teal,
  "SVR": maroon,
)
#let MODEL_ORDER = ("ST-GNN", "MLP", "LSTM", "CVAE", "KNN", "Linear", "SVR")
#let TRUE_PV_COLOR = black

#let ts = data.timestamps.map(parse-dt)
#let true_pv = data.true_pv

// One shared y-axis across every panel (per the figure spec) -- computed
// from the *signal* series only (true PV, every model's q50), deliberately
// excluding both the P10-P90 bands and net load. Including the bands (as an
// earlier version did) let SVR's much wider quantile spread single-handedly
// stretch every panel's shared axis, crushing the other six models' bands
// down to near-invisible -- the bands are supposed to communicate relative
// uncertainty, which this defeated entirely. Bands are still drawn at their
// real width; an unusually wide one (SVR) simply crops at its own panel's
// frame instead of dictating everyone else's scale -- itself an honest
// signal that SVR's uncertainty doesn't fit the same range as the rest.
// Net load is excluded too -- it's not plotted anymore (see file header),
// and including it would reintroduce the negative-axis headroom this whole
// change is meant to remove. Padding is intentionally tight (10.5%, down
// 30% from an earlier 15%) to keep the shared range close to the real data.
#let global-ylim = {
  let vals = true_pv
  for name in MODEL_ORDER {
    vals = vals + data.models.at(name).pv_q50
  }
  vals = vals.filter(v => v != none)
  let lo = calc.min(0.0, ..vals)
  let hi = calc.max(..vals)
  let pad = (hi - lo) * 0.105
  (lo - pad, hi + pad)
}

// One panel: true PV always; predicted q50 + P10-P90 band only when
// `model` is given (none => the truth-only reference panel).
#let panel(title-text, model: none) = {
  let plots = (
    lq.plot(
      ts, true_pv, stroke: (paint: TRUE_PV_COLOR, thickness: 0.9pt),
      mark: none, label: [True PV], step: end,
    ),
  )
  if model != none {
    let m = data.models.at(model)
    let color = MODEL_COLORS.at(model)
    plots += (
      lq.fill-between(
        ts, m.pv_qlo, y2: m.pv_qhi,
        fill: color.transparentize(80%), stroke: none,
        label: [P10-P90], step: end,
      ),
      lq.plot(
        ts, m.pv_q50, stroke: (paint: color, thickness: 0.8pt),
        mark: none, label: model, step: end,
      ),
    )
  }
  lq.diagram(
    title: [#text(size: 0.8em)[#title-text]],
    xlabel: none,
    ylabel: [Power (W)],
    ylim: global-ylim,
    // 0% + length -> full panel footprint (axes/labels/title included), not
    // just the data area -- lilaq's default `width`/`height` only sizes the
    // data area, which let axis-label overhead balloon the real figure well
    // past the intended page-fit target.
    width: 0% + 4.3cm,
    height: 0% + 3.0cm,
    legend: none,
    ..plots
  )
}

#align(center)[
  #text(size: 0.85em)[household #data.household · #data.date]
]
#v(0.1em)

#align(center, grid(
  columns: 4, rows: 2, column-gutter: 0.2em, row-gutter: 0.25em,
  panel([Truth]),
  ..MODEL_ORDER.map(name => panel(name, model: name)),
))

#v(0.15em)
#align(center, grid(
  columns: 4, column-gutter: 0.6em, row-gutter: 0.2em, align: horizon,
  line(length: 1.3em, stroke: (paint: TRUE_PV_COLOR, thickness: 1.4pt)), text(size: 0.7em)[True PV],
  box(width: 1.3em, height: 0.8em, fill: gray.transparentize(80%)), text(size: 0.7em)[P10-P90 band],
))
