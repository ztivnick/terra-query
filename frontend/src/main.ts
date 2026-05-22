import "./styles.css";

import { absThumbUrl, getAoi, healthz, postSearch } from "./api";
import type { SearchHit } from "./api";
import { TerraMap } from "./map";

const appEl = document.getElementById("app");
if (!appEl) throw new Error("missing #app element");

// z-[1000] on #detail beats Leaflet's pane z-indexes (~200-800) which
// would otherwise paint over the panel.
appEl.innerHTML = `
  <div class="flex h-full w-full">
    <aside class="flex w-96 shrink-0 flex-col border-r border-stone-300 bg-white">
      <header class="border-b border-stone-200 px-4 py-3">
        <h1 class="text-lg font-semibold tracking-tight">terra-query</h1>
        <p id="health" class="mt-1 text-xs text-stone-500">connecting...</p>
      </header>
      <form id="query-form" class="border-b border-stone-200 px-4 py-3">
        <label for="q" class="block text-xs font-medium text-stone-600">query</label>
        <input
          id="q"
          name="q"
          type="text"
          autocomplete="off"
          placeholder="search aerial imagery"
          class="mt-1 w-full rounded border border-stone-300 px-2 py-1.5 text-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
        />
        <div class="mt-2 flex items-center justify-between">
          <label class="inline-flex items-center gap-1.5 text-xs text-stone-600">
            <input id="inside-aoi" type="checkbox" class="rounded border-stone-300" />
            inside AOI only
          </label>
          <button
            id="query-submit"
            type="submit"
            class="rounded bg-sky-600 px-3 py-1 text-xs font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-stone-300"
          >search</button>
        </div>
        <p id="query-msg" class="mt-2 min-h-[1em] text-xs text-stone-500"></p>
      </form>
      <ol id="results" class="flex-1 divide-y divide-stone-100 overflow-y-auto"></ol>
    </aside>
    <main class="relative flex-1 min-w-0">
      <div id="map"></div>
      <div
        id="detail"
        class="absolute right-3 top-3 z-[1000] hidden w-72 rounded-md border border-stone-300 bg-white/95 p-3 text-sm shadow-md backdrop-blur"
      ></div>
    </main>
  </div>
`;

const healthEl = document.getElementById("health") as HTMLParagraphElement;
const formEl = document.getElementById("query-form") as HTMLFormElement;
const inputEl = document.getElementById("q") as HTMLInputElement;
const submitEl = document.getElementById("query-submit") as HTMLButtonElement;
const insideAoiEl = document.getElementById("inside-aoi") as HTMLInputElement;
const msgEl = document.getElementById("query-msg") as HTMLParagraphElement;
const resultsEl = document.getElementById("results") as HTMLOListElement;
const detailEl = document.getElementById("detail") as HTMLDivElement;
const mapEl = document.getElementById("map") as HTMLDivElement;

const tmap = new TerraMap(mapEl);

// re-measure the map when the viewport changes
window.addEventListener("resize", () => tmap.invalidateSize());

let currentHits: SearchHit[] = [];
let searchInFlight = false;

bootstrap().catch((err) => {
  console.error(err);
  healthEl.textContent = `error: ${err instanceof Error ? err.message : String(err)}`;
  healthEl.classList.add("text-red-600");
});

async function bootstrap() {
  const h = await healthz();
  healthEl.textContent = `model=${h.model_id}  experiment=${h.experiment_id}  dim=${h.embed_dim}`;
  const aoi = await getAoi();
  tmap.drawAoi(aoi);
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (searchInFlight) return;
  const text = inputEl.value.trim();
  if (!text) return;

  searchInFlight = true;
  submitEl.disabled = true;
  msgEl.classList.remove("text-red-600");
  msgEl.textContent = "searching...";
  resultsEl.innerHTML = "";
  hideDetail();
  tmap.clearResults();

  try {
    const res = await postSearch({
      text,
      top_k: 10,
      inside_aoi_only: insideAoiEl.checked,
    });
    currentHits = res.results;
    renderResults(res.results);
    tmap.drawResults(res.results, { onHitSelect: selectHit });
    msgEl.textContent = `${res.results.length} result${res.results.length === 1 ? "" : "s"} for "${text}"`;
  } catch (err) {
    console.error(err);
    msgEl.textContent = err instanceof Error ? err.message : String(err);
    msgEl.classList.add("text-red-600");
  } finally {
    searchInFlight = false;
    submitEl.disabled = false;
  }
});

function renderResults(hits: SearchHit[]) {
  if (hits.length === 0) {
    resultsEl.innerHTML = `<li class="px-4 py-3 text-sm text-stone-500">no results</li>`;
    return;
  }
  resultsEl.innerHTML = hits
    .map((h, i) => {
      const [lat, lon] = h.center_wgs84;
      return `
        <li
          data-idx="${i}"
          class="cursor-pointer px-4 py-2 hover:bg-sky-50 aria-current:bg-sky-50 aria-current:ring-1 aria-current:ring-inset aria-current:ring-sky-300"
        >
          <div class="flex items-baseline justify-between">
            <span class="text-sm font-medium">#${i + 1} ${escapeHtml(h.chip_location_id)}</span>
            <span class="text-xs text-stone-500">cos ${h.score.toFixed(3)}</span>
          </div>
          <div class="text-xs text-stone-500">
            cycle ${escapeHtml(h.winning_cycle)} &middot; ${lat.toFixed(4)}, ${lon.toFixed(4)}
            ${h.inside_aoi ? "" : ' &middot; <span class="text-amber-600">outside AOI</span>'}
          </div>
        </li>`;
    })
    .join("");
  resultsEl.querySelectorAll<HTMLLIElement>("li[data-idx]").forEach((li) => {
    li.addEventListener("click", () => {
      const idx = Number(li.dataset.idx);
      const hit = currentHits[idx];
      if (hit) selectHit(hit, idx);
    });
  });
}

function selectHit(hit: SearchHit, index: number) {
  tmap.focusHit(index);
  showDetail(hit, index);
  // mark the matching list item; scroll it into view if offscreen
  resultsEl.querySelectorAll<HTMLLIElement>("li[data-idx]").forEach((li) => {
    if (Number(li.dataset.idx) === index) {
      li.setAttribute("aria-current", "true");
      li.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } else {
      li.removeAttribute("aria-current");
    }
  });
}

function showDetail(hit: SearchHit, index: number) {
  const [lat, lon] = hit.center_wgs84;
  detailEl.replaceChildren();
  const html = `
    <button
      id="detail-close"
      type="button"
      aria-label="close"
      class="absolute right-2 top-2 cursor-pointer rounded p-1 leading-none text-stone-400 hover:bg-stone-100 hover:text-stone-700"
    >&times;</button>
    <div class="flex items-baseline justify-between pr-6">
      <span class="text-sm font-semibold">#${index + 1} ${escapeHtml(hit.chip_location_id)}</span>
      <span class="text-xs text-stone-500">cos ${hit.score.toFixed(3)}</span>
    </div>
    <div class="mt-1 text-xs text-stone-600">
      cycle ${escapeHtml(hit.winning_cycle)} &middot; ${lat.toFixed(5)}, ${lon.toFixed(5)}
      ${hit.inside_aoi ? "" : ' &middot; <span class="text-amber-600">outside AOI</span>'}
    </div>
    <img
      src="${escapeAttr(absThumbUrl(hit.thumbnail_url))}"
      alt="chip ${escapeAttr(hit.chip_location_id)} (${escapeAttr(hit.winning_cycle)})"
      width="256" height="256"
      class="mt-2 aspect-square w-full rounded border border-stone-200 bg-stone-100 object-cover"
    />
  `;
  detailEl.insertAdjacentHTML("afterbegin", html);
  detailEl.classList.remove("hidden");
  detailEl
    .querySelector<HTMLButtonElement>("#detail-close")
    ?.addEventListener("click", hideDetail);
}

function hideDetail() {
  detailEl.classList.add("hidden");
  detailEl.replaceChildren();
}

// defensive escapers for strings spliced into the DOM via insertAdjacentHTML
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttr(s: string): string {
  return escapeHtml(s);
}
