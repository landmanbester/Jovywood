
import click


@click.group()
def cli():
    pass

from jove.workers import concat
from jove.workers import extract
from jove.workers import fit
from jove.workers import interp
from jove.workers import cube2fits