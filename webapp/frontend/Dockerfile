FROM node:20-alpine AS build
WORKDIR /app
COPY webapp/frontend/package.json webapp/frontend/package-lock.json* ./
RUN npm ci
COPY webapp/frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY webapp/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
