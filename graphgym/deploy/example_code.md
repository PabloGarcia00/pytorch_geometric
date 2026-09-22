# Earn-E Datahub API — Reference Notes

## Authentication

API key is stored in a `.env` file under `API_KEY`.

```js
headers: {
  'Content-Type': 'application/json',
  'hub-api-key': VALID_API_KEY,
}
```

## 4.1 GET devices

Returns a collection of inverter device ID and P1 dongle ID.

- **P1Dongle ID** is represented as a UUID, e.g. `3d6a243c-8d41-4762-800e-886a0775`
- **Inverter ID** is represented as, e.g. `6a355db6ddbbc60f37d806ca`

**Endpoint:** `GET https://datahub.earn-e.com/api/v1/historic/getdevices`

Example response:

```json
[
  [
    "20ac8d22-5313-4026-b352-4231d65a",
    "6a355db6ddbbc60f37d806ca",
    "6a67a1a0b4073084ef6982b1"
  ],
  [
    "3d6a243c-8d41-4762-800e-886a0775",
    "6a357bbfddbbc60f37d84824"
  ]
]
```

## 4.2 POST graph

Returns an aggregated result of devices posted as input.

**Endpoint:** `POST https://datahub.earn-e.com/api/v1/historic/graph`

```js
const res = await apiClient.post('/api/v1/historic/graph', {
  deviceIds: [
    { id: "dd7939c6-b7ae-4488-b5ae-94b379e9" },
    { id: "6a70a03b163e1dc794bbdbc0" }
  ],
  start: 1787868000000,
  end: 1787954399999,
  samples: 1440
})
```

Example response:

```js
graph: [
  {
    timeStamp: '2026-08-27T22:00:00.000Z',
    avgImportKw: 0.048,
    avgExportKw: 0,
    totalGas: 4668.859863,
    interpolatedPowerKw: null,
    netImportKw: 0.048,
    netExportKw: 0,
    selfConsumptionKw: 0,
    buildingLoad: 48
  },
  {
    timeStamp: '2026-08-27T22:01:00.000Z',
    avgImportKw: 0.048,
    avgExportKw: 0,
    totalGas: 4668.859863,
    interpolatedPowerKw: null,
    netImportKw: 0.048,
    netExportKw: 0,
    selfConsumptionKw: 0,
    buildingLoad: 48
  },
  {
    timeStamp: '2026-08-27T22:02:00.000Z',
    avgImportKw: 0.048,
    avgExportKw: 0,
    totalGas: 4668.859863,
    interpolatedPowerKw: null,
    netImportKw: 0.048,
    netExportKw: 0,
    selfConsumptionKw: 0,
    buildingLoad: 48
  }
]
```

## 4.3 POST rawsolargraph

Returns raw solar graph data, without linear interpolation.

**Endpoint:** `POST https://datahub.earn-e.com/api/v1/historic/rawsolargraph`

> Max delta between start and end timestamp should not exceed 24 hrs.

```js
const res = await apiClient.post('/api/v1/historic/rawsolargraph', {
  deviceIds: [
    { id: "6a70a03b163e1dc794bbdbc0" }
  ],
  start: 1787868000000,
  end: 1787954399999,
  samples: 1440
})
```

Example response:

```js
graph: [
  { timeStamp: '2026-08-28T05:35:00.000Z', powerOutputKw: 0 },
  { timeStamp: '2026-08-28T05:42:00.000Z', powerOutputKw: 0.01 },
  { timeStamp: '2026-08-28T05:50:00.000Z', powerOutputKw: 0.03 },
  { timeStamp: '2026-08-28T05:57:00.000Z', powerOutputKw: 0.06 },
  { timeStamp: '2026-08-28T06:04:00.000Z', powerOutputKw: 0.08 },
  { timeStamp: '2026-08-28T06:12:00.000Z', powerOutputKw: 0.09 },
  { timeStamp: '2026-08-28T06:18:00.000Z', powerOutputKw: 0.11 },
  { timeStamp: '2026-08-28T06:26:00.000Z', powerOutputKw: 0.1 },
  { timeStamp: '2026-08-28T06:33:00.000Z', powerOutputKw: 0.09 },
  { timeStamp: '2026-08-28T06:39:00.000Z', powerOutputKw: 0.1 },
  { timeStamp: '2026-08-28T06:46:00.000Z', powerOutputKw: 0.07 },
  { timeStamp: '2026-08-28T06:53:00.000Z', powerOutputKw: 0.11 },
  { timeStamp: '2026-08-28T07:01:00.000Z', powerOutputKw: 0.15 },
  { timeStamp: '2026-08-28T07:08:00.000Z', powerOutputKw: 0.13 },
  { timeStamp: '2026-08-28T07:15:00.000Z', powerOutputKw: 0.1 },
  { timeStamp: '2026-08-28T07:22:00.000Z', powerOutputKw: 0.13 },
]
```
