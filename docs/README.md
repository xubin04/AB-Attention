# AB-Attention project page

Public site: https://xubin04.github.io/AB-Attention/

GitHub Pages serves `docs/` from `main`. Pushing changes to this directory
updates the site automatically. No build step or npm dependencies are required.
Edit `index.html` for content, `static/css/index.css` for layout, and
`static/js/index.js` for the video controls. Preview locally with
`python3 -m http.server 8000 --directory docs` from the repository root.

## Content

- Title, authors, affiliations, abstract, PDF, and Table 1 match the technical
  report as of 2026-10-15 (paper source commit `9499824`).
- Videos are original Wan2.1-14B outputs, with matched dense and sparse samples
  for Penguin prompts 038 (magpie), 490 (motorcycle), and 521 (mushrooms).
  Gallery-only prompts are 107 (maples) and 448 (house). PSNR is rounded to two
  decimals from the recorded full-video metrics. MP4 indices are moved to the
  start for web playback without re-encoding the video. Clips were generated on GH200;
  the paper's reported end-to-end speedups were measured on H20.
- Add the arXiv link and update BibTeX when an identifier is available.
- The page accent matches the report's `reportaccent` (`#BB0819`), with
  `cayenne` (`#941100`) for hover states. Keep the favicon and social preview
  in the same palette.
- Author links use verified personal homepages where available, otherwise
  the supplied OpenReview profiles. Affiliations link to their official
  websites; InfRec links to its Cardinal AI site.

## Attribution

Adapted from [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
and [Nerfies](https://nerfies.github.io/). The adapted template and styling use
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Bulma uses the MIT license (see `static/css/BULMA-LICENSE.txt`).
Inter and Manrope are self-hosted under the SIL Open Font License;
their notices are included in `static/fonts/`.
These template/font licenses do not alter the licenses of the research media,
paper, or AB-Attention code.
