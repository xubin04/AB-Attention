const samples = {
  magpie: {
    prompt: 'A magpie takes off with a twig in its beak, in anime style.',
    psnr: { 'ab-attention': '30.51', 'dfs-attn': '26.64', svg2: '27.53' },
  },
  motorcycle: {
    prompt: 'Kamen Rider was speeding down the road on his motorcycle.',
    psnr: { 'ab-attention': '30.37', 'dfs-attn': '26.75', svg2: '26.53' },
  },
  mushrooms: {
    prompt: 'Behind the rubber tree, a cluster of wild mushrooms is sprouting quietly. Taken with a time-lapse shot, the process of mushrooms emerging from the soil and gradually unfolding their umbrella-shaped caps is captured.',
    psnr: { 'ab-attention': '31.71', 'dfs-attn': '27.97', svg2: '27.79' },
  },
};

// The hero gallery plays only while visible. Reduced-motion users start paused.
const gallery = [...document.querySelectorAll('.gallery-video')];
const galleryToggle = document.querySelector('#gallery-toggle');
let galleryEnabled = !matchMedia('(prefers-reduced-motion: reduce)').matches;
const visibleGallery = new Set();

function updateGalleryLabel() {
  const playing = gallery.some(video => !video.paused);
  galleryToggle.textContent = playing ? 'Pause gallery' : 'Play gallery';
  galleryToggle.setAttribute('aria-pressed', String(playing));
}

const galleryObserver = new IntersectionObserver(entries => {
  for (const { target, isIntersecting } of entries) {
    if (isIntersecting) visibleGallery.add(target);
    else visibleGallery.delete(target);
    if (isIntersecting && galleryEnabled && !document.hidden) {
      target.play().catch(updateGalleryLabel);
    } else target.pause();
  }
}, { threshold: 0.2 });

for (const video of gallery) {
  video.controls = false;
  galleryObserver.observe(video);
  video.addEventListener('play', updateGalleryLabel);
  video.addEventListener('pause', updateGalleryLabel);
}
galleryToggle.hidden = false;
galleryToggle.addEventListener('click', () => {
  galleryEnabled = !gallery.some(video => !video.paused);
  for (const video of gallery) {
    if (galleryEnabled && visibleGallery.has(video)) video.play().catch(updateGalleryLabel);
    else video.pause();
  }
});

// One transport controls the four paired videos.
const figures = [...document.querySelectorAll('.comparison-grid figure')];
const videos = figures.map(figure => figure.querySelector('video'));
const master = videos[0];
const playButton = document.querySelector('#comparison-play');
const seek = document.querySelector('#comparison-seek');
const timeLabel = document.querySelector('#comparison-time');
let playing = false;
let playRequest = 0;

function stopComparison() {
  playRequest += 1;
  playing = false;
  videos.forEach(video => video.pause());
  playButton.disabled = false;
  playButton.textContent = 'Play all';
  playButton.setAttribute('aria-pressed', 'false');
}

function ready(video) {
  if (video.readyState >= 3) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const finish = error => {
      clearTimeout(timeout);
      video.removeEventListener('canplay', loaded);
      video.removeEventListener('error', failed);
      error ? reject(error) : resolve();
    };
    const loaded = () => finish();
    const failed = () => finish(new Error('Video could not be loaded.'));
    const timeout = setTimeout(failed, 60000);
    video.addEventListener('canplay', loaded, { once: true });
    video.addEventListener('error', failed, { once: true });
    video.preload = 'auto';
    if (video.networkState === HTMLMediaElement.NETWORK_EMPTY) video.load();
  });
}

async function playComparison() {
  const request = ++playRequest;
  playButton.disabled = true;
  playButton.textContent = 'Loading…';
  try {
    await Promise.all(videos.map(ready));
    if (request !== playRequest) return;
    const start = master.ended || master.currentTime >= master.duration - 0.05 ? 0 : master.currentTime;
    videos.forEach(video => { video.currentTime = start; });
    await Promise.all(videos.map(video => video.play()));
    if (request !== playRequest) return;
    playing = true;
    playButton.textContent = 'Pause all';
    playButton.setAttribute('aria-pressed', 'true');
  } catch {
    if (request === playRequest) {
      stopComparison();
      playButton.textContent = 'Retry play';
    }
  } finally {
    if (request === playRequest) playButton.disabled = false;
  }
}

function updateTimeline() {
  if (!Number.isFinite(master.duration)) return;
  seek.value = String(100 * master.currentTime / master.duration);
  const timestamp = value => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
  timeLabel.textContent = `${timestamp(master.currentTime)} / ${timestamp(master.duration)}`;
}

for (const video of videos) {
  video.controls = false;
  video.preload = 'metadata';
}
document.querySelector('.comparison-controls').hidden = false;
document.querySelector('.sample-selector').hidden = false;
playButton.addEventListener('click', () => playing ? stopComparison() : playComparison());
document.querySelector('#comparison-restart').addEventListener('click', () => {
  videos.forEach(video => { if (video.readyState) video.currentTime = 0; });
  seek.value = '0';
  updateTimeline();
});
seek.addEventListener('input', () => {
  const fraction = Number(seek.value) / 100;
  videos.forEach(video => { if (Number.isFinite(video.duration)) video.currentTime = fraction * video.duration; });
  updateTimeline();
});
master.addEventListener('loadedmetadata', updateTimeline);
master.addEventListener('timeupdate', () => {
  updateTimeline();
  if (playing) {
    for (const video of videos.slice(1)) {
      if (Math.abs(video.currentTime - master.currentTime) > 0.12) video.currentTime = master.currentTime;
    }
  }
});
master.addEventListener('ended', stopComparison);

for (const button of document.querySelectorAll('[data-sample]')) {
  button.addEventListener('click', () => {
    if (button.getAttribute('aria-pressed') === 'true') return;
    stopComparison();
    const name = button.dataset.sample;
    const sample = samples[name];
    document.querySelectorAll('[data-sample]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    document.querySelector('#comparison-prompt').textContent = `“${sample.prompt}”`;
    for (const figure of figures) {
      const method = figure.dataset.method;
      const video = figure.querySelector('video');
      video.poster = `static/images/${name}-${method}.webp`;
      video.querySelector('source').src = `static/videos/${name}-${method}.mp4`;
      video.load();
      figure.querySelector('.video-metric').textContent = method === 'dense' ? 'Full attention' : `PSNR ${sample.psnr[method]} dB`;
    }
    seek.value = '0';
    timeLabel.textContent = '0:00 / 0:03';
  });
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    gallery.forEach(video => video.pause());
    stopComparison();
  } else if (galleryEnabled) {
    visibleGallery.forEach(video => video.play().catch(updateGalleryLabel));
  }
});

const copyButton = document.querySelector('#copy-bibtex');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  const citation = document.querySelector('#citation');
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = 'Citation copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
