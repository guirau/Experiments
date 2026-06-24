const NAMES: Record<string, string> = {
  thong_sala: "Thong Sala", ban_tai: "Ban Tai", ban_kai: "Ban Kai", haad_rin: "Haad Rin",
  srithanu: "Sri Thanu", chaloklum: "Chaloklum", mae_haad: "Mae Haad", hin_kong: "Hin Kong",
  woktum: "Woktum", haad_yao: "Haad Yao", haad_salad: "Haad Salad", haad_son: "Haad Son",
  thong_nai_pan: "Thong Nai Pan", bottle_beach: "Bottle Beach", than_sadet: "Than Sadet",
  haad_yuan_tien: "Haad Yuan/Tien", madeua_wan: "Madeua Wan", plai_laem: "Plai Laem",
  other: "Other", unknown: "Unknown",
};
export function areaName(slug: string | null): string {
  if (!slug) return "Unknown";
  return NAMES[slug] ?? slug;
}
