import { cp, rm, mkdir } from 'node:fs/promises';
const src = new URL('./site/', import.meta.url);
const out = new URL('./dist/', import.meta.url);
await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });
await cp(src, out, { recursive: true });
console.log('Pixel-Ops static Vercel bundle created in dist/');
