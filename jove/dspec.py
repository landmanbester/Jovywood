
import click


@click.group()
def cli():
    pass

from jove.workers import ift_smooth2
from jove.workers import gpr_smooth
