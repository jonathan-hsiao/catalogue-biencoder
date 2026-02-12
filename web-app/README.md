# Product Embedding Constellations Web App

Interactive 3D visualization of product embeddings (~49K points) with dark space theme, category stars, hover tooltips, neighbor comparison, and search/filter capabilities.

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment variable** (optional):
   ```bash
   cp .env.local.example .env.local
   # Edit .env.local: set NEXT_PUBLIC_ASSET_BASE_URL if using CDN
   ```

   **Note:** `NEXT_PUBLIC_ASSET_BASE_URL` is **build-time** (baked into the bundle). For local development, leave it empty (`""`). For production with CDN, set it at build time (e.g., in Vercel environment variables).

3. **Place data files:**
   - Copy `web_data/v1/` directory (from `prep_web_data.py` output) to `public/web_data/v1/`
   - Extract `thumbs_150px_public.zip` to `public/thumbs/` (should have `thumbs/train/` and `thumbs/test/` subdirectories)

4. **Run development server:**
   ```bash
   npm run dev
   ```

   Open [http://localhost:3000](http://localhost:3000)

## Build & Deploy

```bash
npm run build
npm start
```

### CDN Deployment

For production, host `web_data/` and `thumbs/` on a CDN/object storage (Cloudflare R2, S3, etc.):

1. **Set CORS:** Your CDN must allow GET requests from your app origin:
   ```
   Access-Control-Allow-Origin: <your-app-domain>
   ```
   Or use `*` for public access.

2. **Set `NEXT_PUBLIC_ASSET_BASE_URL`** at build time to your CDN URL:
   ```
   NEXT_PUBLIC_ASSET_BASE_URL=https://cdn.example.com
   ```

3. **Build:** The base URL is baked into the bundle, so rebuild if you change it.

### Runtime Config (Optional)

If you need the same build to work with multiple origins, use runtime config:

1. Create `/public/config.json`:
   ```json
   { "assetBaseUrl": "" }
   ```

2. Fetch it at app load and use instead of `NEXT_PUBLIC_ASSET_BASE_URL`.

## Data Structure

The app expects:
- `public/web_data/v1/manifest.json` - Schema-validated manifest
- `public/web_data/v1/data/*.bin` - Binary files (positions, labels, neighbors)
- `public/web_data/v1/data/meta/meta_*.json` - Sharded product metadata
- `public/web_data/v1/categories/*.json` - Category lists
- `public/web_data/v1/categories/*.bin` - Star data
- `public/thumbs/{train|test}/{idx}.webp` - Thumbnail images

## Features

- **3D Point Cloud:** ~49K products rendered as points with custom shader
- **Category Stars:** Coarse category centroids colored by top-level category
- **Hover Tooltips:** Product/star details on hover
- **Click Selection:** Select product to see neighbors; select star to see gravity lines
- **Search:** Full-text search by title, brand, category (after meta preload)
- **Filters:** All/Train/Test split, ambiguous-only toggle
- **Gravity Lines:** Lines from selected star to member products (sampled)

## Performance

- **Initial load:** Binaries load first → point cloud renders → meta preloads in background
- **Meta parsing:** Uses `requestIdleCallback` to avoid main-thread hitches
- **Hover picking:** Throttled to `requestAnimationFrame`, skipped while camera moves
- **Filtering:** Split filter uses shader uniform (no buffer rewrite); search updates `aVisible` attribute

## Browser Support

Requires WebGL2 support. Tested on Chrome, Firefox, Safari.
