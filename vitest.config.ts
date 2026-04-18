import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    snapshotFormat: {
      maxOutputLength: Number.MAX_SAFE_INTEGER,
    },
  },
});
