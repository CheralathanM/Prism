"""In-car navigation extension: destination changes mid-request, handled by the same kernel.

The driver asks for a destination, then changes their mind while the route is still being
planned. The kernel supersedes the stale route, never starts navigation to the old
destination, and starts navigation exactly once to the corrected one.
"""
