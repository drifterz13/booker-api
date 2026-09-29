"""Baseline for the original book upload table.

Existing databases must verify this schema before stamping 0001.
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "book",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("object_key", sa.String(), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column(
            "status",
            sa.Enum("UPLOADING", "UPLOADED", "FAILED", name="bookstatus"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("object_key"),
    )


def downgrade() -> None:
    op.drop_table("book")
    sa.Enum(name="bookstatus").drop(op.get_bind(), checkfirst=True)
