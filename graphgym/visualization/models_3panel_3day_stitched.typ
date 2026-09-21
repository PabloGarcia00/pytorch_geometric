// models_3panel_3day_stitched.typ -- frontend for
// export_3panel_3day_stitched.py's JSON payload. 3 stacked panels, one
// model per panel (CVAE / MLP / KNN by default). Each panel plots a
// stitched 96x3-step profile: 3 back-to-back day-segments, each segment
// from its own (household, date) pair -- NOT one continuous real
// timeline, so the x-axis is a stitched step index (0-287) rather than
// real datetimes. The same 3 segments (same households, same dates) are
// reused identically across all 3 panels -- only the model differs.
// Segment (household, date) labels are attached directly above each
// panel's own plot (not one shared header far above the stack), and
// segment-boundary dividers are drawn solid/dark/thick with an explicit
// "day break" callout so it's unambiguous these separate different
// (household, date) windows, not a continuous trajectory.
//
// Usage: typst compile --input json=data/models_3panel_3day_stitched.json models_3panel_3day_stitched.typ out.pdf
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: (x: 8mm, y: 8mm))
#set text(font: "New Computer Modern", size: 9pt)
#set par(leading: 0.5em)

#let data = json(sys.inputs.at("json", default: "data/models_3panel_3day_stitched_test.json"))
#let n = data.steps_per_day

// Fixed real-vs-predicted two-color scheme (not per-model identity) --
// each panel is already labeled by model name in its title, so distinct
// per-model colors aren't needed to tell panels apart.
#let TRUE_PV_COLOR = blue
#let PRED_PV_COLOR = orange


#let x = range(data.panels.at(0).true_pv.len())

// Per-panel segment-label strip: "household H · date D" centered above
// each of the 3 segments, placed directly above THIS panel's own plot
// (repeated per panel, not a single shared header) so the labelling lives
// with the plot it describes.
#let segment-labels() = align(center, grid(
  columns: data.segments.len(),
  column-gutter: 0.4em,
  ..data.segments.map(seg => text(size: 0.7em, fill: gray.darken(30%))[household #seg.household · #seg.date])
))

#let panel(p) = {
  segment-labels()
  v(0.05em)
  lq.diagram(
    xlabel: [Time (15-minute)],
    ylabel: [PV (W)],
    width: 11cm,
    height: 3.6cm,
    legend: none,
    // Major grid every 24 steps == quarter day at 15-min cadence (96 steps/day / 4).
    xaxis: (tick-distance: 24),
    // Draw order matters here: uncertainty band on the bottom, then Real,
    // then the predicted median on top -- so the (semi-transparent but
    // still layered) band never sits on top of and visually obscures the
    // true PV line, and the model's own prediction is always the topmost,
    // most legible series.
    lq.fill-between(
      x, p.pv_qlo, y2: p.pv_qhi,
      fill: PRED_PV_COLOR.lighten(80%), stroke: none,
      label: [P10-P90], step: end,
    ),
    lq.plot(
      x, p.true_pv, stroke: (paint: TRUE_PV_COLOR, thickness: 1pt),
      mark: none, label: [Real], step: end,
    ),
    lq.plot(
      x, p.pv_q50, stroke: (paint: PRED_PV_COLOR, thickness: 1pt),
      mark: none, label: p.model, step: end,
    ),
  )
}

#align(center)[
  #text(size: 1.05em)[#data.config]
]
#v(0.1em)
#align(center, text(size: 0.65em, fill: gray)[#data.segment_rule])
#v(0.3em)

#align(center, stack(
  spacing: 0.55em,
  ..data.panels.map(p => panel(p))
))

#v(0.4em)
#align(center, grid(
  columns: 6, column-gutter: 0.6em, row-gutter: 0.3em, align: horizon,
  line(length: 1.3em, stroke: (paint: TRUE_PV_COLOR, thickness: 1.2pt)), text(size: 0.8em)[Real],
  box(width: 1.3em, height: 0.8em, fill: gray.transparentize(80%)), text(size: 0.8em)[P10-P90 band],
))
