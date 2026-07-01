export const PROPERTY_TYPES = ["house", "villa", "bungalow", "apartment", "studio", "room", "unknown"];
export const SEASONS = ["low", "high", "full_year", "unknown"];
export const CONFIDENCES = ["high", "medium", "low"];
export const LANGUAGES = ["en", "th", "mixed", "other"];
export const AMENITY_FIELDS: { field: keyof import("./types").Listing; label: string }[] = [
  { field: "has_aircon", label: "Aircon" }, { field: "has_wifi", label: "Wifi" },
  { field: "has_pool", label: "Pool" }, { field: "has_kitchen", label: "Kitchen" },
  { field: "has_parking", label: "Parking" }, { field: "furnished", label: "Furnished" },
  { field: "pet_friendly", label: "Pet friendly" }, { field: "sea_view", label: "Sea view" },
  { field: "has_workspace", label: "Workspace" }, { field: "has_terrace", label: "Terrace" },
];
export const AREA_ENUM = ["thong_sala","ban_tai","ban_kai","haad_rin","srithanu","chaloklum","mae_haad","hin_kong","woktum","haad_yao","haad_salad","haad_son","thong_nai_pan","bottle_beach","than_sadet","haad_yuan_tien","madeua_wan","plai_laem","other","unknown"];
