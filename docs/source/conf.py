project = "readcon-ops"
copyright = "2026, Rohit Goswami"
author = "Rohit Goswami"
release = "0.1.1"

extensions = []
templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "shibuya"
html_title = "readcon-ops"
html_static_path = ["_static"]

html_context = {
    "source_type": "github",
    "source_user": "lode-org",
    "source_repo": "readcon-ops",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}

html_theme_options = {
    "github_url": "https://github.com/lode-org/readcon-ops",
    "accent_color": "indigo",
    "dark_code": True,
    "show_ai_links": False,
    "globaltoc_expand_depth": 1,
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {"title": "eOn", "url": "https://eondocs.org"},
                {"title": "rgpot", "url": "https://rgpot.rgoswami.me"},
                {"title": "rgmin", "url": "https://omnipotentrpc.github.io/rgmin"},
                {"title": "rgsaddle", "url": "https://omnipotentrpc.github.io/rgsaddle"},
                {"title": "cpmdc", "url": "https://cpmdc.rgoswami.me"},
                {"title": "rgpycrumbs", "url": "https://rgpycrumbs.rgoswami.me"},
                {"title": "minimage", "url": "https://lode-org.github.io/minimage"},
                {"title": "linkcell", "url": "https://docs.rs/linkcell"},
                {"title": "readcon", "url": "https://lode-org.github.io/readcon-core"},
                {"title": "readcon-db", "url": "https://lode-org.github.io/readcon-db"},
                {"title": "readcon-ops", "url": "https://github.com/lode-org/readcon-ops"},
            ],
        },
    ],
}
