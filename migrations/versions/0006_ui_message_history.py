"""Use UI messages as the single stored conversation history."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("conversationmessage", "model_messages")


def downgrade() -> None:
    op.add_column(
        "conversationmessage",
        sa.Column("model_messages", postgresql.JSONB(), nullable=True),
    )
