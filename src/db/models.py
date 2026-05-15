"""
SQLAlchemy ORM models for the NeuroSleep project.

Schema rationale
----------------
Three tables capture the experiment lifecycle:

- experiments:  one row per modeling run (RF / SVM / CNN).
                Stores hyperparameters as JSON for flexibility across models.
- metrics:      one row per (experiment, metric_name).
                Long-format makes it easy to query, compare, and aggregate.
- predictions:  one row per (experiment, test_epoch).
                Allows post-hoc analysis like per-subject error inspection.

Design decisions:
- We do not store raw signals in SQL (they're large and binary). Those live in
  the data/processed directory or object storage in a production deployment.
- Hyperparameters are stored as JSONB to avoid schema churn when a new model
  introduces new parameters.
- Metrics are long-format rather than wide-format to support arbitrary metric
  sets across models.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Experiment(Base):
    """One modeling run."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(64), nullable=False, index=True)
    dataset = Column(String(64), nullable=False, default="sleep-edf")
    n_train_epochs = Column(Integer, nullable=True)
    n_test_epochs = Column(Integer, nullable=True)
    n_features = Column(Integer, nullable=True)
    train_subjects = Column(JSONB, nullable=True)
    test_subjects = Column(JSONB, nullable=True)
    hyperparameters = Column(JSONB, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    metrics = relationship(
        "Metric", back_populates="experiment", cascade="all, delete-orphan"
    )
    predictions = relationship(
        "Prediction", back_populates="experiment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Experiment(id={self.id}, model={self.model_name!r}, "
            f"created_at={self.created_at:%Y-%m-%d %H:%M})>"
        )


class Metric(Base):
    """A single (name, value) pair attached to an experiment."""

    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(
        Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name = Column(String(64), nullable=False, index=True)
    value = Column(Float, nullable=False)
    split = Column(String(32), nullable=False, default="test")

    experiment = relationship("Experiment", back_populates="metrics")

    def __repr__(self) -> str:
        return f"<Metric({self.name}={self.value:.4f}, exp={self.experiment_id})>"


class Prediction(Base):
    """One predicted label for one test epoch."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(
        Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_id = Column(Integer, nullable=False, index=True)
    true_label = Column(String(8), nullable=False)
    predicted_label = Column(String(8), nullable=False)

    experiment = relationship("Experiment", back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"<Prediction(exp={self.experiment_id}, "
            f"true={self.true_label}, pred={self.predicted_label})>"
        )
