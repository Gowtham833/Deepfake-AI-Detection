# scripts/generate_pwa_icons.py
"""Generate creative, high-resolution PWA icons and SVG logo for Deepfake AI Detection."""

import math
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter


def create_vector_logo_svg(path: Path):
    """Generate modern SVG logo for the header and branding."""
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0a0f1d" />
      <stop offset="50%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#030712" />
    </linearGradient>
    <linearGradient id="borderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="50%" stop-color="#818cf8" />
      <stop offset="100%" stop-color="#c084fc" />
    </linearGradient>
    <linearGradient id="cyanGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22d3ee" />
      <stop offset="100%" stop-color="#0284c7" />
    </linearGradient>
    <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f43f5e" />
      <stop offset="50%" stop-color="#a855f7" />
      <stop offset="100%" stop-color="#6366f1" />
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="12" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Background Shield / Squircle -->
  <rect x="24" y="24" width="464" height="464" rx="100" fill="url(#bgGrad)" stroke="url(#borderGrad)" stroke-width="8" filter="url(#glow)" />

  <!-- Radar / Scan Rings -->
  <circle cx="256" cy="245" r="170" fill="none" stroke="#1e293b" stroke-width="2" stroke-dasharray="8 8" />
  <circle cx="256" cy="245" r="125" fill="none" stroke="#334155" stroke-width="2" stroke-dasharray="6 6" />

  <!-- Biometric Corner Brackets -->
  <path d="M 120 160 L 120 125 A 15 15 0 0 1 135 110 L 170 110" fill="none" stroke="#38bdf8" stroke-width="6" stroke-linecap="round" />
  <path d="M 392 160 L 392 125 A 15 15 0 0 0 377 110 L 342 110" fill="none" stroke="#a855f7" stroke-width="6" stroke-linecap="round" />
  <path d="M 120 330 L 120 365 A 15 15 0 0 0 135 380 L 170 380" fill="none" stroke="#38bdf8" stroke-width="6" stroke-linecap="round" />
  <path d="M 392 330 L 392 365 A 15 15 0 0 1 377 380 L 342 380" fill="none" stroke="#a855f7" stroke-width="6" stroke-linecap="round" />

  <!-- Left Side: Real Face Silhouette (Smooth, Cyan Glow) -->
  <path d="M 256 120 C 210 120 170 155 170 205 C 170 245 185 275 205 305 C 220 327 240 348 256 360 L 256 120 Z" 
        fill="url(#cyanGrad)" opacity="0.85" filter="url(#glow)" />

  <!-- Right Side: Deepfake Digital Mesh / Synthetic Face (Neural Wireframe) -->
  <!-- Wireframe polygons -->
  <polygon points="256,120 310,135 256,180" fill="#a855f7" opacity="0.3" stroke="#c084fc" stroke-width="2" />
  <polygon points="310,135 342,185 256,180" fill="#6366f1" opacity="0.25" stroke="#818cf8" stroke-width="2" />
  <polygon points="256,180 342,185 320,245" fill="#ec4899" opacity="0.3" stroke="#f43f5e" stroke-width="2" />
  <polygon points="256,180 320,245 256,255" fill="#a855f7" opacity="0.4" stroke="#c084fc" stroke-width="2" />
  <polygon points="256,255 320,245 300,305" fill="#6366f1" opacity="0.3" stroke="#818cf8" stroke-width="2" />
  <polygon points="256,255 300,305 256,315" fill="#a855f7" opacity="0.35" stroke="#c084fc" stroke-width="2" />
  <polygon points="256,315 300,305 256,360" fill="#ec4899" opacity="0.25" stroke="#f43f5e" stroke-width="2" />

  <!-- Neural Nodes (Circles) -->
  <circle cx="310" cy="135" r="5" fill="#38bdf8" />
  <circle cx="342" cy="185" r="6" fill="#f43f5e" />
  <circle cx="320" cy="245" r="6" fill="#c084fc" />
  <circle cx="300" cy="305" r="5" fill="#38bdf8" />
  <circle cx="256" cy="180" r="5" fill="#ffffff" />
  <circle cx="256" cy="255" r="5" fill="#ffffff" />
  <circle cx="256" cy="315" r="5" fill="#ffffff" />

  <!-- Center Biometric Laser Scan Line -->
  <line x1="140" y1="245" x2="372" y2="245" stroke="#00f2fe" stroke-width="4" stroke-linecap="round" filter="url(#glow)" />
  <polygon points="256,236 265,245 256,254 247,245" fill="#ffffff" filter="url(#glow)" />

  <!-- Verification Shield Icon at Bottom -->
  <circle cx="256" cy="410" r="32" fill="#0f172a" stroke="#22d3ee" stroke-width="3" filter="url(#glow)" />
  <!-- AI Checkmark in Shield -->
  <path d="M 244 410 L 253 419 L 270 399" fill="none" stroke="#22d3ee" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
</svg>"""
    path.write_text(svg_content, encoding="utf-8")
    print(f"Created SVG: {path}")


def draw_master_icon(size: int = 1024, padding_ratio: float = 0.08) -> Image.Image:
    """Draw a master high-resolution icon using PIL with anti-aliasing."""
    # Create canvas with dark gradient background
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = int(size * padding_ratio)
    box = [pad, pad, size - pad, size - pad]
    radius = int(size * 0.22)

    # 1. Dark Rounded Badge with subtle border
    # Background gradient approximation
    draw.rounded_rectangle(box, radius=radius, fill=(10, 15, 29, 255), outline=(56, 189, 248, 255), width=int(size * 0.016))

    center_x = size // 2
    center_y = int(size * 0.47)

    # 2. Concentric Target / Biometric Rings
    r_outer = int(size * 0.33)
    r_inner = int(size * 0.24)
    draw.ellipse([center_x - r_outer, center_y - r_outer, center_x + r_outer, center_y + r_outer], outline=(30, 41, 59, 255), width=max(2, size // 256))
    draw.ellipse([center_x - r_inner, center_y - r_inner, center_x + r_inner, center_y + r_inner], outline=(51, 65, 85, 255), width=max(2, size // 256))

    # 3. Corner Brackets (Biometric Viewfinder)
    bracket_len = int(size * 0.09)
    b_pad = int(size * 0.22)
    b_w = max(4, int(size * 0.014))

    # Top-Left Bracket (Cyan)
    draw.line([(b_pad, b_pad + bracket_len), (b_pad, b_pad), (b_pad + bracket_len, b_pad)], fill=(34, 211, 238, 255), width=b_w)
    # Top-Right Bracket (Purple)
    draw.line([(size - b_pad, b_pad + bracket_len), (size - b_pad, b_pad), (size - b_pad - bracket_len, b_pad)], fill=(192, 132, 252, 255), width=b_w)
    # Bottom-Left Bracket (Cyan)
    draw.line([(b_pad, size - b_pad - bracket_len), (b_pad, size - b_pad), (b_pad + bracket_len, size - b_pad)], fill=(34, 211, 238, 255), width=b_w)
    # Bottom-Right Bracket (Purple)
    draw.line([(size - b_pad, size - b_pad - bracket_len), (size - b_pad, size - b_pad), (size - b_pad - bracket_len, size - b_pad)], fill=(192, 132, 252, 255), width=b_w)

    # 4. Human Face Contour (Left Half)
    # Organic smooth polygon
    head_top = int(size * 0.22)
    chin_bottom = int(size * 0.70)
    left_cheek = int(size * 0.33)
    left_jaw = int(size * 0.39)

    left_face_points = [
        (center_x, head_top),
        (int(size * 0.42), int(size * 0.24)),
        (int(size * 0.36), int(size * 0.31)),
        (left_cheek, int(size * 0.42)),
        (left_cheek, int(size * 0.52)),
        (left_jaw, int(size * 0.62)),
        (center_x, chin_bottom),
    ]
    draw.polygon(left_face_points, fill=(2, 132, 199, 230), outline=(56, 189, 248, 255))

    # 5. Synthetic / Neural Network Wireframe (Right Half)
    right_nodes = [
        (center_x, head_top),
        (int(size * 0.59), int(size * 0.25)),
        (int(size * 0.66), int(size * 0.36)),
        (center_x, int(size * 0.35)),
        (int(size * 0.63), int(size * 0.48)),
        (center_x, int(size * 0.50)),
        (int(size * 0.58), int(size * 0.60)),
        (center_x, int(size * 0.61)),
        (center_x, chin_bottom),
    ]

    triangles = [
        (right_nodes[0], right_nodes[1], right_nodes[3], (168, 85, 247, 160)),
        (right_nodes[1], right_nodes[2], right_nodes[3], (99, 102, 241, 150)),
        (right_nodes[3], right_nodes[2], right_nodes[4], (236, 72, 153, 170)),
        (right_nodes[3], right_nodes[4], right_nodes[5], (168, 85, 247, 180)),
        (right_nodes[5], right_nodes[4], right_nodes[6], (99, 102, 241, 160)),
        (right_nodes[5], right_nodes[6], right_nodes[7], (168, 85, 247, 170)),
        (right_nodes[7], right_nodes[6], right_nodes[8], (236, 72, 153, 150)),
    ]

    for p1, p2, p3, fill_col in triangles:
        draw.polygon([p1, p2, p3], fill=fill_col, outline=(192, 132, 252, 255))

    # Neural node dots
    dot_r = max(4, int(size * 0.012))
    for pt in right_nodes:
        draw.ellipse([pt[0] - dot_r, pt[1] - dot_r, pt[0] + dot_r, pt[1] + dot_r], fill=(255, 255, 255, 255), outline=(56, 189, 248, 255))

    # 6. Center Biometric Scanning Laser Beam
    laser_y = int(size * 0.48)
    laser_left = int(size * 0.25)
    laser_right = int(size * 0.75)
    draw.line([(laser_left, laser_y), (laser_right, laser_y)], fill=(34, 211, 238, 255), width=max(3, int(size * 0.008)))

    # Diamond at center of scan beam
    d_size = int(size * 0.02)
    diamond = [
        (center_x, laser_y - d_size),
        (center_x + d_size, laser_y),
        (center_x, laser_y + d_size),
        (center_x - d_size, laser_y),
    ]
    draw.polygon(diamond, fill=(255, 255, 255, 255), outline=(34, 211, 238, 255))

    # 7. Verification Shield at Bottom
    shield_y = int(size * 0.81)
    s_rad = int(size * 0.075)
    draw.ellipse([center_x - s_rad, shield_y - s_rad, center_x + s_rad, shield_y + s_rad], fill=(15, 23, 42, 255), outline=(34, 211, 238, 255), width=max(3, int(size * 0.009)))

    # Checkmark inside Shield
    chk_w = max(3, int(size * 0.009))
    p_chk1 = (center_x - int(s_rad * 0.45), shield_y)
    p_chk2 = (center_x - int(s_rad * 0.05), shield_y + int(s_rad * 0.40))
    p_chk3 = (center_x + int(s_rad * 0.45), shield_y - int(s_rad * 0.35))
    draw.line([p_chk1, p_chk2, p_chk3], fill=(34, 211, 238, 255), width=chk_w, joint="curve")

    return img


def main():
    root = Path(__file__).resolve().parent.parent
    app_static = root / "app" / "static"
    frontend = root / "frontend"

    # SVG logo
    create_vector_logo_svg(app_static / "logo.svg")
    create_vector_logo_svg(frontend / "logo.svg")

    # Generate Standard Icons (padding 0.06)
    master = draw_master_icon(size=1024, padding_ratio=0.06)

    # Generate Maskable Icons for Android (padding 0.16 for circular safe zone)
    maskable_master = draw_master_icon(size=1024, padding_ratio=0.18)

    sizes = [
        ("icon-512.png", 512, master),
        ("icon-192.png", 192, master),
        ("icon-maskable-512.png", 512, maskable_master),
        ("icon-maskable-192.png", 192, maskable_master),
        ("favicon.png", 64, master),
    ]

    target_dirs = [app_static / "icons", frontend / "icons"]
    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for name, sz, source_img in sizes:
            resized = source_img.resize((sz, sz), Image.Resampling.LANCZOS)
            out_file = d / name
            resized.save(out_file, "PNG", optimize=True)
            print(f"Saved: {out_file} ({sz}x{sz})")

    # Also save favicon in static root
    fav = master.resize((64, 64), Image.Resampling.LANCZOS)
    fav.save(app_static / "favicon.png", "PNG")
    fav.save(frontend / "favicon.png", "PNG")
    print("Generated all PWA icons successfully!")


if __name__ == "__main__":
    main()
