// physics_profile.typ -- frontend half of the JSON -> Typst pipeline, physics
// mode. One model per figure (always), four vertically-stacked panels --
// Real, no mask/loss, mask only, mask+loss -- each showing exactly ONE
// series against the physical export floor (PV must stay above this --
// below it is physically impossible), rather than overlaying every tier
// together. This keeps each series independently legible even when one
// tier's line would otherwise occlude the others at this scale (e.g.
// mask+loss spiking to several times the real trajectory, as
// physics_weight=0.3 does for several models) -- and the "Real" panel
// makes visible that even the ground truth sometimes dips below its own
// export floor (a metering/measurement artifact, not a model failure).
// All panels share one normalization (the real trajectory's own peak) so
// panel-to-panel comparison stays valid.
//
// Usage: typst compile --input json=data/physics_MLP.json physics_profile.typ out.pdf
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: (x: 8mm, y: 8mm))
#set text(font: "New Computer Modern", size: 9pt)
#set par(leading: 0.5em)

#let data = json(sys.inputs.at("json", default: "data/physics_test.json"))

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

#let nanmax(arr) = {
  let valid = arr.filter(v => v != none)
  if valid.len() == 0 { return 1.0 }
  valid.fold(valid.at(0), (acc, v) => if v > acc { v } else { acc })
}

// Fixed series -> (display label, color) so the same series always gets
// the same color across every model's figure, unlike models_profile.typ's
// arbitrary per-model palette assignment. "real" is included as its own
// panel (not just overlaid in every other panel) since the user asked for
// four panels, each series plotted against the export line alone.
#let SERIES_STYLE = (
  real: ("Real", black),
  no_mask_no_loss: ("No mask/loss", orange),
  mask_only: ("Mask only", blue),
  mask_and_loss: ("Mask + loss", green.darken(20%)),
)
#let SERIES_ORDER = ("real", "no_mask_no_loss", "mask_only", "mask_and_loss")

#let ts = data.timestamps.map(parse-dt)
#let true-max = calc.max(nanmax(data.true_pv), 1e-6)
#let norm(arr) = arr.map(v => if v == none { none } else { calc.max(v / true-max, 0.0) })

#let series-values(key) = if key == "real" { data.true_pv } else { data.tiers.at(key).pv_q50 }

#align(center)[
  #text(size: 1em)[#data.model #h(1em) #text(size: 0.85em, fill: gray)[household #data.household · #data.date]]
  #v(0.3em)
  #stack(
    dir: ttb, spacing: 0.4em,
    ..SERIES_ORDER.enumerate().map(((i, key)) => {
      let (label, color) = SERIES_STYLE.at(key)
      lq.diagram(
        title: [#text(size: 0.85em)[#label]],
        xlabel: if i == SERIES_ORDER.len() - 1 { [Time] } else { none },
        ylabel: [Watt (rel.)],
        width: 10cm,
        height: 2.3cm,
        legend: none,
        lq.plot(
          ts, norm(data.export_true),
          stroke: (paint: gray.darken(30%), thickness: 0.7pt),
          mark: none, label: [Export (PV floor)], step: end,
        ),
        lq.plot(
          ts, norm(series-values(key)),
          stroke: (paint: color, thickness: 0.7pt), mark: none, label: label, step: end,
        ),
      )
    })
  )
]

#v(0.4em)
#align(center, grid(
  columns: 10,
  column-gutter: 0.6em, row-gutter: 0.3em, align: horizon,
  line(length: 1.3em, stroke: (paint: gray.darken(30%), thickness: 1pt)), text(size: 0.8em)[Export floor],
  ..SERIES_ORDER.map(key => {
    let (label, color) = SERIES_STYLE.at(key)
    (line(length: 1.3em, stroke: (paint: color, thickness: 1.2pt)), text(size: 0.8em)[#label])
  }).flatten()
))
