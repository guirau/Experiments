import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "KP House Finder", description: "Personal Koh Phangan rental dashboard" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
