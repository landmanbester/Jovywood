
import click


@click.group()
def cli():
    pass

from jove.workers import ift_smooth
from jove.workers import gpr_smooth