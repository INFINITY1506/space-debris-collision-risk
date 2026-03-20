# Professional UI Redesign Plan

## Goal
Transform the UI from "AI-generated" look into a professional mission-control aesthetic (Bloomberg/Palantir/SpaceX style).

## Key Changes

### Design Direction
- **No emojis** — replace all 13 emojis with simple SVG icons or text
- **No Orbitron font** — Inter only, clean weights (400/500/600)
- **No star field background** — solid dark navy
- **No gradient text** — solid colors only
- **No glow effects** — clean borders, subtle shadows
- **1 accent color** — blue #3B82F6 only. Semantic colors (red/amber/green) only for risk levels
- **Minimal animations** — only fade-in + spinner. Remove orbit/sweep/pulse

### Files to Edit (in order)
1. `index.css` — New design tokens, remove star field, remove decorative animations
2. `LoadingSpinner.tsx` — Replace radar with simple spinner, remove emojis, StatCard drops `icon` prop
3. `App.tsx` — Clean header (SVG logo, no emoji), plain text hero, no gradient, emoji-free feature cards
4. `SatelliteCard.tsx` — Remove orbit animation, uniform data colors
5. `SearchBar.tsx` — Remove emojis from chips
6. `ResultsTable.tsx` — Remove pulse animations, consistent colors
7. `GlobeView.tsx` — Clean overlays, remove backdrop-filter
8. `index.html` — Replace emoji favicon with SVG
