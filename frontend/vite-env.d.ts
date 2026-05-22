/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TERRA_QUERY_API_URL?: string;
  readonly VITE_TERRA_QUERY_TILE_URL?: string;
  readonly VITE_TERRA_QUERY_TILE_ATTRIBUTION?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
