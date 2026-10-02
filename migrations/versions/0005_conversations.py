"""Store public book conversations and their visible and agent messages."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["book_id"], ["book.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_conversation_book_updated_at", "conversation", ["book_id", "updated_at"]
    )
    op.create_table(
        "conversationmessage",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column(
            "role",
            sa.Enum("USER", "ASSISTANT", name="conversationrole"),
            nullable=False,
        ),
        sa.Column("ui_message", postgresql.JSONB(), nullable=False),
        sa.Column("model_messages", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversation.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "conversation_id", "position", name="uq_conversationmessage_position"
        ),
        sa.CheckConstraint("position >= 0", name="ck_conversationmessage_position"),
    )


def downgrade() -> None:
    op.drop_table("conversationmessage")
    sa.Enum(name="conversationrole").drop(op.get_bind(), checkfirst=True)
    op.drop_table("conversation")
