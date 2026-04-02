# Base stage for dependencies
FROM node:20-slim AS base
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable

# Development stage
FROM base AS deps
WORKDIR /app

# Copy root workspace config and lockfile
COPY pnpm-lock.yaml pnpm-workspace.yaml package.json ./

# Copy all packages (required for workspace dependency resolution)
COPY packages ./packages
COPY apps/web/package.json ./apps/web/package.json

# Install dependencies (only for web app)
RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --frozen-lockfile

# Build stage
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/apps/web/node_modules ./apps/web/node_modules
COPY --from=deps /app/packages ./packages
COPY . .

# Set environment variables for build
ENV NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

# Build the web app
RUN pnpm --filter @resound-studio/web build

# Production stage
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

# Create a system user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy necessary files
COPY --from=builder /app/apps/web/public ./apps/web/public
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next ./apps/web/.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/apps/web/node_modules ./apps/web/node_modules
COPY --from=builder /app/apps/web/package.json ./apps/web/package.json
COPY --from=builder /app/package.json ./package.json

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

# Command to run (Next.js start)
WORKDIR /app/apps/web
CMD ["pnpm", "start"]
