// models_profile.typ -- frontend half of the JSON -> Typst pipeline.
// Renders one panel, every model's best-seed PV Q50 overlaid on the real
// trajectory, normalized to the real trajectory's own peak (so over/under-
// shoot stays visible -- each model isn't independently rescaled to its own
// peak, which would hide magnitude errors entirely).
//
// Usage: typst compile --input json=data/models_dual_wx_off.json models_profile.typ out.pdf
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: (x: 8mm, y: 8mm))
#set text(font: "New Computer Modern", size: 9pt)
#set par(leading: 0.5em)

#let data = json(sys.inputs.at("json", default: "data/models_test.json"))

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

#let COLORS = (orange, blue, green.darken(20%), purple, red.lighten(20%), teal, maroon)

#let ts = data.timestamps.map(parse-dt)
#let true-max = calc.max(nanmax(data.true_pv), 1e-6)
#let norm(arr) = arr.map(v => if v == none { none } else { calc.max(v / true-max, 0.0) })

#let model-names = data.models.keys().sorted()

#align(center)[
  #lq.diagram(
    title: [#data.config #h(1em) #text(size: 0.85em, fill: gray)[household #data.household · #data.date]],
    xlabel: [Time],
    ylabel: [Watt (relative)],
    width: 10cm,
    height: 4.2cm,
    legend: none,
    lq.plot(
      ts, norm(data.true_pv), stroke: (paint: black, thickness: 1pt),
      mark: none, label: [Real], step: end,
    ),
    ..model-names.enumerate().map(((i, name)) => {
      lq.plot(
        ts, norm(data.models.at(name).pv_q50),
        stroke: (paint: COLORS.at(calc.rem(i, COLORS.len())), thickness: 0.7pt),
        mark: none, label: name, step: end,
      )
    })
  )
]

#v(0.4em)
#align(center, grid(
  columns: (model-names.len() + 1) * 2,
  column-gutter: 0.6em, row-gutter: 0.3em, align: horizon,
  line(length: 1.3em, stroke: (paint: black, thickness: 1.2pt)), text(size: 0.8em)[Real],
  ..model-names.enumerate().map(((i, name)) => (
    line(length: 1.3em, stroke: (paint: COLORS.at(calc.rem(i, COLORS.len())), thickness: 1.2pt)),
    text(size: 0.8em)[#name],
  )).flatten()
))
