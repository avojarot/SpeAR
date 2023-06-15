from sqlalchemy import Column, ForeignKey, Integer, Unicode
from sqlalchemy.orm import as_declarative, declared_attr


@as_declarative()
class AbstractModel:
    @classmethod
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()


class DataFormat(AbstractModel):
    id = Column(Integer, autoincrement=True)
    name = Column(Unicode)


class ExportFormat(AbstractModel):
    id = Column(Integer, autoincrement=True)
    name = Column(Unicode)


class User(AbstractModel):
    id = Column(Integer, autoincrement=True)
    name = Column(Unicode)
