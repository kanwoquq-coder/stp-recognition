"""Route factory wrapper.

The quote helper is injected into the implementation module so generated media
and report URLs are safe for Chinese filenames as well as ASCII filenames.
"""

from urllib.parse import quote

from app import routes_impl


routes_impl.quote = quote
create_router = routes_impl.create_router

__all__ = ["create_router"]
