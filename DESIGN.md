---
name: Clinical Precision AI
colors:
  surface: '#f4fbf4'
  surface-dim: '#d4dcd5'
  surface-bright: '#f4fbf4'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eef6ee'
  surface-container: '#e8f0e9'
  surface-container-high: '#e3eae3'
  surface-container-highest: '#dde4dd'
  on-surface: '#161d19'
  on-surface-variant: '#3c4a42'
  inverse-surface: '#2b322d'
  inverse-on-surface: '#ebf3eb'
  outline: '#6c7a71'
  outline-variant: '#bbcabf'
  surface-tint: '#006c49'
  primary: '#006c49'
  on-primary: '#ffffff'
  primary-container: '#10b981'
  on-primary-container: '#00422b'
  inverse-primary: '#4edea3'
  secondary: '#0051d5'
  on-secondary: '#ffffff'
  secondary-container: '#316bf3'
  on-secondary-container: '#fefcff'
  tertiary: '#a43a3a'
  on-tertiary: '#ffffff'
  tertiary-container: '#fc7c78'
  on-tertiary-container: '#711419'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#6ffbbe'
  primary-fixed-dim: '#4edea3'
  on-primary-fixed: '#002113'
  on-primary-fixed-variant: '#005236'
  secondary-fixed: '#dbe1ff'
  secondary-fixed-dim: '#b4c5ff'
  on-secondary-fixed: '#00174b'
  on-secondary-fixed-variant: '#003ea8'
  tertiary-fixed: '#ffdad7'
  tertiary-fixed-dim: '#ffb3af'
  on-tertiary-fixed: '#410005'
  on-tertiary-fixed-variant: '#842225'
  background: '#f4fbf4'
  on-background: '#161d19'
  surface-variant: '#dde4dd'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  data-mono:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  gutter: 20px
  margin: 24px
  max_width: 1440px
---

## Brand & Style
The design system is engineered for the high-stakes environment of life sciences and medical diagnostics. It balances the sterile reliability of traditional healthcare with the cutting-edge intelligence of modern AI. 

The aesthetic is **Modern Corporate** with **Minimalist** leanings. It prioritizes clarity, cognitive ease, and trust. The visual language avoids decorative flourishes, opting instead for functional precision. The emotional response should be one of "calm authority"—reassuring the practitioner that the data is accurate, the AI is a reliable partner, and the platform is an advanced medical instrument rather than a consumer app.

## Colors
The palette is rooted in clinical standards. **Vitality Green** serves as the primary driver for "go" actions and healthy status indicators. **Clinical Blue** is utilized for systemic navigation and interactive elements, anchoring the platform in technological trust. 

The background utilizes **Soft Hospital White** to reduce eye strain during long clinical shifts, while **Deep Charcoal** provides the necessary contrast ratio for WCAG AAA compliance on critical medical text. **Soft Mint** is reserved exclusively for non-interactive backgrounds of positive diagnostic results to provide a subtle, calming visual cue.

## Typography
This design system employs a dual-font strategy. **Inter** handles all user interface elements, labels, and narrative clinical notes to ensure maximum readability and a modern, accessible feel. 

**JetBrains Mono** is reserved for technical telemetry, genomic sequences, lab values, and raw AI confidence scores. This distinction helps practitioners immediately categorize information: Inter for human-readable context, JetBrains Mono for objective, machine-generated data. Use `label-caps` for table headers and section categorizers to maintain an organized, tabular feel typical of professional EMRs.

## Layout & Spacing
The layout follows a **Fixed Grid** model on desktop to ensure data density remains predictable for clinical review. It uses a 12-column grid with a 1440px max-width container. 

A strict 4px soft-grid system governs all padding and margins. In data-heavy views, use "Compact" spacing (8px between elements), while in diagnostic dashboards, use "Spacious" padding (24px) to allow for clear visual isolation of critical charts or imagery. On mobile, the layout collapses to a single column with 16px side margins, prioritizing the most recent clinical alerts at the top of the scroll.

## Elevation & Depth
Hierarchy is established through **Tonal Layers** and **Ambient Shadows**. Instead of heavy dropshadows, the design system uses a two-tier elevation system:
1.  **Level 0 (Base):** The #F8FAFC surface.
2.  **Level 1 (Cards):** White (#FFFFFF) surfaces with a 1px border (#E2E8F0) and a very soft, diffused shadow (0px 2px 4px rgba(30, 41, 59, 0.05)).
3.  **Level 2 (Modals/Popovers):** White surfaces with a slightly deeper shadow (0px 10px 15px rgba(30, 41, 59, 0.1)) to indicate temporary interaction.

Avoid all glows, blurs, or neon effects. The depth should feel physical and grounded, like paper charts stacked on a clean desk.

## Shapes
The design system uses a **Soft** shape language. Standard UI elements like input fields, buttons, and small cards use a 0.25rem (4px) radius. Larger dashboard containers use a 0.5rem (8px) radius. This conservative rounding maintains a professional, "instrument-like" precision while removing the aggressive sharpness of pure 90-degree angles, making the high-tech AI components feel more approachable and modern.

## Components
-   **Buttons:** Primary actions use a solid **Vitality Green** background with white text. Secondary actions use a **Clinical Blue** outline with a transparent background. High-contrast is mandatory; never use ghost buttons for critical clinical operations.
-   **Cards:** The primary container for AI insights. Cards must include a 1px #E2E8F0 border. Headers within cards should use the `label-caps` style for clarity.
-   **Data Tables:** Use alternating row stripes (Zebra striping) using #F1F5F9 for the "off" row. Use **JetBrains Mono** for numerical values.
-   **Status Chips:** Small, pill-shaped indicators. "Normal" results use **Soft Mint** background with **Vitality Green** text. "Critical" results use a light red tint with **Warning Red** text.
-   **Input Fields:** Use a 1px border. On focus, the border transitions to **Clinical Blue** with a 2px outer ring of 10% opacity blue.
-   **AI Indicators:** Any insight generated by the AI should be marked with a subtle 2px vertical accent bar of **Clinical Blue** on the left side of the component to distinguish it from manual human entries.