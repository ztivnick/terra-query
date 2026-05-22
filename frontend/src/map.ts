// Leaflet map setup + helpers for drawing the AOI and result pins / footprints.

import L from "leaflet";
import type { SearchHit, AoiResponse } from "./api";

// Tile sources are env-driven so providers can swap without a code edit.
// Set VITE_TERRA_QUERY_LABELS_URL="" to disable the labels overlay.
const TILE_URL =
  import.meta.env.VITE_TERRA_QUERY_TILE_URL ??
  "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const TILE_ATTR =
  import.meta.env.VITE_TERRA_QUERY_TILE_ATTRIBUTION ??
  "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community";
const LABELS_URL =
  import.meta.env.VITE_TERRA_QUERY_LABELS_URL ??
  "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}";
const LABELS_ATTR =
  import.meta.env.VITE_TERRA_QUERY_LABELS_ATTRIBUTION ??
  "Labels &copy; Esri";

// initial map view; replaced as soon as /aoi loads + fitBounds runs.
const DEFAULT_CENTER = L.latLng(0, 0);

export interface PinHandlers {
  onHitSelect: (hit: SearchHit, index: number) => void;
}

interface HitLayer {
  polygon: L.Polygon;
  marker: L.CircleMarker;
}

export class TerraMap {
  readonly map: L.Map;
  private aoiLayer: L.GeoJSON | null = null;
  private resultLayer: L.LayerGroup;
  private hitLayers: HitLayer[] = [];
  private selectedIndex: number | null = null;

  constructor(el: HTMLElement) {
    this.map = L.map(el, {
      center: DEFAULT_CENTER,
      zoom: 2,
      preferCanvas: true,
    });
    L.tileLayer(TILE_URL, {
      attribution: TILE_ATTR,
      maxZoom: 19,
      keepBuffer: 4,
    }).addTo(this.map);

    if (LABELS_URL) {
      // drawn after the base layer -> paints on top
      L.tileLayer(LABELS_URL, {
        attribution: LABELS_ATTR,
        maxZoom: 19,
        keepBuffer: 4,
        pane: "tilePane",
      }).addTo(this.map);
    }

    this.resultLayer = L.layerGroup().addTo(this.map);
  }

  /** Call when the container's size changes (e.g. window resize). */
  invalidateSize(): void {
    this.map.invalidateSize();
  }

  drawAoi(aoi: AoiResponse): void {
    if (this.aoiLayer) this.aoiLayer.remove();
    this.aoiLayer = L.geoJSON(aoi.geojson as GeoJSON.GeoJsonObject, {
      style: {
        color: "#0ea5e9",
        weight: 2,
        fillOpacity: 0.04,
        interactive: false,
      },
    }).addTo(this.map);
    const bounds = this.aoiLayer.getBounds();
    if (bounds.isValid()) {
      this.map.fitBounds(bounds, { padding: [16, 16] });
    }
  }

  clearResults(): void {
    this.resultLayer.clearLayers();
    this.hitLayers = [];
    this.selectedIndex = null;
  }

  drawResults(hits: SearchHit[], handlers: PinHandlers): void {
    this.clearResults();
    const container = this.map.getContainer();

    // canvas mode doesn't auto-apply .leaflet-interactive's pointer cursor,
    // so wire it per layer
    const onOver = () => { container.style.cursor = "pointer"; };
    const onOut = () => { container.style.cursor = ""; };

    hits.forEach((hit, i) => {
      const ring: L.LatLngExpression[] = hit.bbox_wgs84.map(([lat, lon]) => [
        lat,
        lon,
      ]);

      // same select on the rectangle as on the pin
      const polygon = L.polygon(ring, {
        color: "#dc2626",
        weight: 1,
        fillColor: "#dc2626",
        fillOpacity: 0.08,
      })
        .on("click", (ev) => {
          L.DomEvent.stop(ev);
          handlers.onHitSelect(hit, i);
        })
        .on("mouseover", onOver)
        .on("mouseout", onOut)
        .addTo(this.resultLayer);

      const marker = L.circleMarker(hit.center_wgs84 as L.LatLngExpression, {
        radius: rankRadius(i),
        color: "#dc2626",
        weight: 2,
        fillColor: "#fef2f2",
        fillOpacity: 0.95,
      })
        .bindTooltip(`#${i + 1}  cos=${hit.score.toFixed(3)}`, {
          direction: "top",
        })
        .on("click", (ev) => {
          L.DomEvent.stop(ev);
          handlers.onHitSelect(hit, i);
        })
        .on("mouseover", onOver)
        .on("mouseout", onOut)
        .addTo(this.resultLayer);

      this.hitLayers.push({ polygon, marker });
    });
  }

  /** Pan to a hit, open its tooltip, and highlight it. */
  focusHit(index: number): void {
    const layer = this.hitLayers[index];
    if (!layer) return;

    if (this.selectedIndex !== null && this.selectedIndex !== index) {
      const prev = this.hitLayers[this.selectedIndex];
      if (prev) {
        prev.marker.setStyle({
          fillColor: "#fef2f2",
          color: "#dc2626",
          weight: 2,
        });
        prev.polygon.setStyle({ weight: 1, fillOpacity: 0.08 });
      }
    }

    layer.marker.setStyle({
      fillColor: "#fde047",
      color: "#b45309",
      weight: 3,
    });
    layer.marker.bringToFront();
    layer.polygon.setStyle({ weight: 2, fillOpacity: 0.16 });

    this.map.panTo(layer.marker.getLatLng());
    layer.marker.openTooltip();
    this.selectedIndex = index;
  }
}

function rankRadius(rank: number): number {
  return Math.max(5, 11 - rank);
}
