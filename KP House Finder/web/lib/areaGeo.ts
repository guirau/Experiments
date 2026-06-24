// Approximate positions of each canonical Koh Phangan area on a 100x100 SVG canvas
// (x: 0=west→100=east, y: 0=north→100=south). These are stylized, NOT survey-accurate —
// tweak the numbers here to nudge any marker. "other" and "unknown" have no location and
// are shown as chips beside the map instead.
export interface AreaPoint { slug: string; x: number; y: number; }

export const AREA_POINTS: AreaPoint[] = [
  { slug: "chaloklum", x: 44, y: 16 },
  { slug: "bottle_beach", x: 61, y: 16 },
  { slug: "mae_haad", x: 23, y: 25 },
  { slug: "thong_nai_pan", x: 79, y: 30 },
  { slug: "haad_salad", x: 19, y: 35 },
  { slug: "haad_yao", x: 15, y: 45 },
  { slug: "than_sadet", x: 83, y: 48 },
  { slug: "haad_son", x: 22, y: 53 },
  { slug: "srithanu", x: 14, y: 58 },
  { slug: "plai_laem", x: 66, y: 56 },
  { slug: "madeua_wan", x: 82, y: 64 },
  { slug: "hin_kong", x: 26, y: 65 },
  { slug: "woktum", x: 31, y: 73 },
  { slug: "thong_sala", x: 43, y: 77 },
  { slug: "ban_tai", x: 53, y: 82 },
  { slug: "ban_kai", x: 64, y: 82 },
  { slug: "haad_yuan_tien", x: 79, y: 74 },
  { slug: "haad_rin", x: 72, y: 91 },
];

// Stylized island silhouette (rounded blob with a southern point toward Haad Rin).
export const ISLAND_PATH =
  "M50,8 C72,7 90,24 90,46 C90,63 85,75 75,85 C68,92 62,95 54,92 " +
  "C46,96 35,91 29,83 C17,73 10,60 10,42 C10,21 28,9 50,8 Z";

// areas in the canonical enum that aren't placed on the map
export const NON_MAP_AREAS = ["other", "unknown"];
