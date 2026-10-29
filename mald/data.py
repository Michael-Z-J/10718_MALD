"""load the built corpus and apply the rule of what is known at prediction time.

a target paper of iclr year Y can use pool forums whose reviews were public
before the iclr Y deadline (late september of Y-1): iclr and neurips <= Y-1.
"""
from functools import cache

import polars as pl

from mald.paths import BUILD


@cache
def forums():
    return pl.read_parquet(BUILD / "forums.parquet")


@cache
def reviews():
    return pl.read_parquet(BUILD / "reviews.parquet")


@cache
def comments():
    return pl.read_parquet(BUILD / "comments.parquet")


def pool(year):
    """pool forums that a target of iclr `year` can use."""
    f = forums().filter(pl.col("split") == "pool")
    return f.filter(pl.col("year") <= year - 1)


def text(forum):
    return (BUILD / "text" / f"{forum}.txt").read_text()


def held(split):
    """test or val forums with text, and their real reviews."""
    f = forums().filter((pl.col("split") == split) & pl.col("has_text"))
    return f, reviews().filter(pl.col("forum").is_in(f["forum"].to_list()))
