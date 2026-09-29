"""Store the generated first-page thumbnail key."""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("book", sa.Column("thumbnail_key", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("book", "thumbnail_key")
