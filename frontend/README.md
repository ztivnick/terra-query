# frontend/

The web UI for terra-query.

This directory exists as a reserved slot for the front end. Specific
implementation choices - vanilla HTML, a framework, bundler, package
manager - belong to the step that builds the UI, not to this README.
Hosting choices likewise belong to whichever step (well after the MVP)
introduces deployment.

What is committed here: the front-end source.

What is not committed: build output, dependency caches, install
artifacts. The repo `.gitignore` already covers the common offenders
(`node_modules/`, `dist/`, `.next/`, `.turbo/`, `*.tsbuildinfo`).
