// Position of each canonical Koh Phangan area as a percentage of the map image
// (web/assets/map.png): x = 0 (west/left) … 100 (east/right), y = 0 (north/top) … 100
// (south/bottom). Estimated by eye against that map's coastline — approximate, so nudge
// any value here to move a marker. "other"/"unknown" have no location (shown as chips).
export interface AreaPoint { slug: string; x: number; y: number; }

export const AREA_POINTS: AreaPoint[] = [
  { slug: "mae_haad", x: 13, y: 9 },
  { slug: "haad_salad", x: 11, y: 17 },
  { slug: "chaloklum", x: 31, y: 12 },
  { slug: "bottle_beach", x: 43, y: 7 },
  { slug: "haad_yao", x: 8, y: 24 },
  { slug: "haad_son", x: 7, y: 31 },
  { slug: "srithanu", x: 7, y: 38 },
  { slug: "plai_laem", x: 70, y: 26 },
  { slug: "thong_nai_pan", x: 86, y: 37 },
  { slug: "than_sadet", x: 82, y: 52 },
  { slug: "madeua_wan", x: 80, y: 64 },
  { slug: "hin_kong", x: 13, y: 51 },
  { slug: "woktum", x: 20, y: 60 },
  { slug: "thong_sala", x: 33, y: 67 },
  { slug: "ban_tai", x: 48, y: 71 },
  { slug: "ban_kai", x: 58, y: 73 },
  { slug: "haad_yuan_tien", x: 71, y: 80 },
  { slug: "haad_rin", x: 61, y: 88 },
];

// areas in the canonical enum that aren't placed on the map
export const NON_MAP_AREAS = ["other", "unknown"];
