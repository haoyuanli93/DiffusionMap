from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import numpy

from layout import Ui_MainWindow


class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        # Define different variables to hold the data
        self.eigenvectors = None
        self.global_index = None
        self.global_index_map = None
        self.label = None
        self.pattern_number = None
        self.eigenvector_number = None

        # Associate the scene object of the manifold to the corresponding object of the view
        self.scene = QtWidgets.QGraphicsScene()
        self.view = self.graphicsView_2.setScene(self.scene)

        # Info from the General column.
        self.manifold_xaxis = None
        self.manifold_yaxis = None

    def load_eigenvectors(self):
        """
        Load the eigenvectors. In the meantime, initialize some related parameters.
        Notice that here, one should not initialize the label object since when one
        first load the eigenvectors, they may or may not have existing label files.

        :return: None
        """
        file_name, flag_1 = QFileDialog.getOpenFileName(self,
                                                        'Please select the npy array to load.',
                                                        "C:\\Users\\haoyu\\Documents\\Python Scripts\\" +
                                                        'diffusion_map_dev\\fake_eigens.npy')
        if flag_1:
            # Load the numpy array
            self.eigenvectors = numpy.load(file_name)
            # Get the size information of the eigenvector file
            self.eigenvector_number, self.pattern_number = self.eigenvectors.shape

    def show_manifold(self):
        """
        Show the manifold for the specified axis.
        :return: None
        """
        # First retrieve the dimensions to show



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
