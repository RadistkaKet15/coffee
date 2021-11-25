import sys
import sqlite3

from PyQt5.QtWidgets import QWidget, QTableView, QApplication, QTableWidgetItem
from PyQt5 import uic


class Example(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('main.ui', self)

        self.connection = sqlite3.connect("coffee.db")
        res = self.connection.cursor().execute("SELECT * FROM coffee").fetchall()
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setRowCount(0)
        for i, row in enumerate(res):
            self.tableWidget.setRowCount(
                self.tableWidget.rowCount() + 1)
            for j, elem in enumerate(row):
                self.tableWidget.setItem(
                    i, j, QTableWidgetItem(str(elem)))

    def closeEvent(self, event):
        self.connection.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec())
