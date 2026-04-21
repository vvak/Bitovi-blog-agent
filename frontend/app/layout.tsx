import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bitovi Blog Agent",
  description: "Ask questions grounded in Bitovi blog articles.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
