# Prompt templates

Replace `{{PRODUCT}}` with the product name spelled out (write "plus", never
`+` — the symbol parses badly). Everything else is deliberate: the phrases
that fight the model's pull toward glossy stock-ad polish are load-bearing,
not decoration.

## creator-ugc — main talking-head clip (`ugc-testimonial-v1`)

References: `ref_image_0` = locked creator, `ref_image_1` = product.
Duration: allow ~13-14s for a 40-word script (natural speech ≈ 150 wpm).
A 6-8s clip only fits ~20 words — trim the script or render longer and use
product B-roll over the middle while the audio continues.

```
Authentic 9:16 TikTok UGC selfie video, front-facing iPhone camera,
chest-up framing, casual handheld movement with tiny natural micro-shakes,
natural imperfect framing, filmed by a real creator and not a commercial
ad. Use the woman in the first reference image as the subject, preserving
exactly the same face, hairstyle, and overall appearance. Use the product
sachet from the second reference image, clearly recognizable as {{PRODUCT}}.
She is in a bright modern Singapore bathroom or bedroom vanity area with
soft natural daylight, clean but lived-in, everyday creator setting.

She looks directly into the phone camera, friendly, warm, believable,
slightly excited, like she is sharing a genuine recommendation with friends.
In the first second, she already holds the {{PRODUCT}} sachet beside her
face so the product is immediately visible. She naturally gestures with the
sachet while speaking. Clear lip-sync, accurate mouth movement, natural
blinking, subtle head movement, realistic skin texture, no heavy beauty-ad
retouching, no glossy commercial stiffness, no stock-ad posing.

She says clearly, naturally, and conversationally:
"{{SCRIPT}}"

Her delivery should sound like genuine TikTok UGC, not scripted TV
advertising. Keep the tone relatable, enthusiastic, and trustworthy. The
sachet branding should remain visible and legible during the clip. End with
a soft smile while still holding the product near the camera. Photorealistic,
believable, creator-style, high-converting TikTok affiliate feel.
```

## product-insert — supporting beauty shot (`product-demo-v1`)

Duration: 3-4s. Cut over the creator's audio as B-roll.

```
Premium vertical product beauty shot, 9:16, elegant but still suitable for
TikTok affiliate editing. The {{PRODUCT}} box is the hero subject, clearly
recognizable and centered, with the sachet also included if appropriate.
Locked-off or very gently controlled camera with a slow subtle push-in.
Warm golden natural light moves softly across the packaging, creating a
luxurious glow. Fine delicate golden petals or particles drift lightly
through the frame, tasteful and restrained, not fantasy-heavy. Soft blurred
background, premium skincare supplement advertising aesthetic, clean,
refined, aspirational, feminine, modern.

Focus on sharp product visibility, beautiful packaging detail, rich premium
texture, and a calm upscale mood. The final moments settle into a clean hero
frame on the box with elegant stillness. Photorealistic, high-end, tasteful,
premium but not overdone.
```

## Scripts

Best-performing shape: natural hook → product named early → key specifics →
close on ease of use (a soft close converts better than a hard "BUY NOW").

```
Okay wait, I've been taking this once a day and my skin really looks a lot
more fresh and glowy. It's {{PRODUCT}} — stem cells, marine collagen, and
twenty-one patents. Super easy, just one sachet a day.
```

Alternative (problem-first hook):

```
If your skin has been looking tired lately, this is the one I've been
taking. {{PRODUCT}} — one sachet a day, with stem cells, marine collagen,
and twenty-one patents. Honestly super easy to add into your routine.
```

## Rules that matter

- Spell out "plus"; never `+`.
- Name which reference is the person and which is the product, explicitly.
- Always request "clear lip-sync, accurate mouth movement" — and put the
  dialogue in the prompt, so H3 generates speech and mouth shapes together.
  Overlaying separate audio onto unrelated mouth movement never syncs.
- Fight glossiness explicitly: "not a commercial ad", "casual handheld",
  "real creator framing", "natural skin texture", "not over-retouched".
- Product visible within the first second.
- When output drifts brand-commercial, add: *"Make it feel like a real
  TikTok creator recommendation, not a polished studio advertisement."*

## Edit structure

Creator hook → product insert as mid-roll B-roll (audio continues) →
back to creator for the close. On-screen text: one sachet a day /
stem cells + marine collagen / 21 patents / easy daily routine.
Soft CTA: "Try it for yourself" or "Tap to shop".
