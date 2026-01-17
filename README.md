# thomasvd.dev

Personal website built with [Hugo](https://gohugo.io/) and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

## Development

### Prerequisites

- [Hugo](https://gohugo.io/installation/) (v0.128.0+)

### Local development

```bash
hugo server -D
```

Visit http://localhost:1313/

### Creating content

**New blog post:**
```bash
hugo new content/blog/my-post-title.md
```

Edit the file, set `draft: false`, then commit and push.

## Deployment

Site automatically deploys to GitHub Pages on every push to `main` via GitHub Actions.

**Setup** (one-time):
1. Go to repo **Settings** → **Pages**
2. Under **Source**, select **GitHub Actions**

## Structure

```
.
├── .github/workflows/     # GitHub Actions
├── content/              # All content
│   ├── about/
│   ├── projects/
│   └── blog/
├── layouts/              # Theme overrides
├── static/               # Static files
├── themes/PaperMod/      # Theme
└── hugo.toml            # Config
```

## License

Content: All rights reserved  
Code: MIT
