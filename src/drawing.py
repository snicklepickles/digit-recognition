from kivy.app import App
from kivy.graphics import Color, Line
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
import predict


class PaintBrush(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 1)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=22)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class DrawingApp(App):
    def build(self):
        boxlayout = BoxLayout()
        self.painter = PaintBrush()
        clear_button = Button(text='Clear')
        clear_button.bind(on_release=self.clear_canvas)
        save_button = Button(text='Predict')
        save_button.bind(on_release=self.predict_canvas)
        boxlayout.add_widget(self.painter)
        sidebar = BoxLayout(orientation='vertical', size_hint=(0.333, 1))
        sidebar.add_widget(clear_button)
        sidebar.add_widget(save_button)
        self.prediction_text = Label(text='KNN:\nRF:')
        sidebar.add_widget(self.prediction_text)
        boxlayout.add_widget(sidebar)
        return boxlayout

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        self.prediction_text.text = 'KNN:\nRF:'

    def predict_canvas(self, obj):
        filename = 'digit.jpg'
        self.painter.export_as_image().save(filename)
        knn, rf = predict.predict(filename)
        self.prediction_text.text = f'KNN: {knn}\nRF: {rf}'


if __name__ == '__main__':
    DrawingApp().run()
