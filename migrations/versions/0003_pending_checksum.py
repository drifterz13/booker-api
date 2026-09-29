"""Allow checksum verification after accepting a completed upload."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("book", "checksum", existing_type=sa.String(64), nullable=True)


def downgrade() -> None:
    # Requires every pending checksum to be verified before downgrading.
    op.alter_column("book", "checksum", existing_type=sa.String(64), nullable=False)
