# frontend/

Vite + TypeScript + Tailwind + Leaflet. Static build, no runtime.

## Run

```bash
cd frontend
npm install
cp .env.example .env.local   # adjust if needed
npm run dev                  # http://localhost:5173
```

## Build

```bash
npm run build      # type-check + emit dist/
npm run preview    # serve dist/ on :4173
```

`dist/` is plain HTML/CSS/JS. Serves from any static host.

## Configuration

All URLs read from env at build time. See `.env.example`.

| var                                  | what                                       |
|--------------------------------------|--------------------------------------------|
| `VITE_TERRA_QUERY_API_URL`           | backend base URL                           |
| `VITE_TERRA_QUERY_TILE_URL`          | base tile URL template                     |
| `VITE_TERRA_QUERY_TILE_ATTRIBUTION`  | base tile attribution                      |
| `VITE_TERRA_QUERY_LABELS_URL`        | optional labels overlay (`""` disables)    |
| `VITE_TERRA_QUERY_LABELS_ATTRIBUTION`| labels attribution                         |

## Layout

- `src/api.ts` - the only module that knows the backend URL.
- `src/map.ts` - leaflet setup + AOI / result drawing.
- `src/main.ts` - app entry: layout, form, results list.
- `src/styles.css` - tailwind + leaflet imports.
- `index.html` - vite entrypoint.
