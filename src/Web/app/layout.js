import "./globals.css";

export const metadata = { title: "PhotoBoo Gallery" };

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
