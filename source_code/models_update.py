# import pickle5 as pickle
import pickle
import numpy
import math
from tensorflow.keras import backend
from tensorflow.keras.models import Model as Model_Keras
from tensorflow.keras import Input
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint

from prepare_nn_features import split_features

#https://www.e-learn.cn/content/wangluowenzhang/889690
#https://blog.csdn.net/anshuai_aw1/article/details/83586404


class Model(object):

    def __init__(self, train_ratio):
        self.train_ratio = train_ratio
        self.__load_data()

    def evaluate(self):
        if self.train_ratio == 1:
            return 0
        total_sqe = 0
        num_real_test = 0
        for record, sales in zip(self.X_val, self.y_val):
            if sales == 0:
                continue
            guessed_sales = self.guess(record)
            sqe = ((sales - guessed_sales) / sales) ** 2
            total_sqe += sqe
            num_real_test += 1
        result = math.sqrt(total_sqe / num_real_test)
        return result

    def __load_data(self):
        with open('feature_train_data.pickle', 'rb') as f:
            (self.X, self.y) = pickle.load(f)
            self.X = numpy.array(self.X)
            self.y = numpy.array(self.y)
            self.num_records = len(self.X)
            self.train_size = int(self.train_ratio * self.num_records)
            self.test_size = self.num_records - self.train_size
            self.X, self.X_val = self.X[:self.train_size], self.X[self.train_size:]
            self.y, self.y_val = self.y[:self.train_size], self.y[self.train_size:]


class NN_with_EntityEmbedding(Model):

    def __init__(self, train_ratio):
        super().__init__(train_ratio)
        self.build_preprocessor(self.X)
        self.nb_epoch = 20
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = numpy.max(numpy.log(self.y))
        self.min_log_y = numpy.min(numpy.log(self.y))
        self.__build_keras_model()
        self.fit()

    def build_preprocessor(self, X):
        X_list = split_features(X)
        # Google trend de
        self.gt_de_enc = StandardScaler()
        self.gt_de_enc.fit(X_list[32])
        # Google trend state
        self.gt_state_enc = StandardScaler()
        self.gt_state_enc.fit(X_list[33])

    def preprocessing(self, X):
        X_list = split_features(X)
        X_list[32] = self.gt_de_enc.transform(X_list[32])
        X_list[33] = self.gt_state_enc.transform(X_list[33])
        return X_list

    def __build_keras_model(self):
        models_in = []
        models_out = []

        # model_store = Sequential()
        # model_store.add(Embedding(1115, 50, input_length=1))
        # # model_store.add(Reshape(dims=(50,)))
        # model_store.add(Reshape((50,)))
        # models.append(model_store)
        # # models.append(model_store.output)

        model_store_in = Input(shape=(1,))
        models_in.append(model_store_in)
        model_store_out = Embedding(1115, 50, input_length=1)(model_store_in)
        model_store_out = Reshape(target_shape=(50,))(model_store_out)
        models_out.append(model_store_out)
        model_store = Model_Keras(model_store_in, model_store_out)

        # model_dow = Sequential()
        # model_dow.add(Embedding(7, 6, input_length=1))
        # # model_dow.add(Reshape(dims=(6,)))
        # model_dow.add(Reshape((6,)))
        # models.append(model_dow)
        # # models.append(model_dow.output)

        model_dow_in = Input(shape=(1,))
        models_in.append(model_dow_in)
        model_dow_out = Embedding(7, 6, input_length=1)(model_dow_in)
        model_dow_out = Reshape(target_shape=(6,))(model_dow_out)
        models_out.append(model_dow_out)
        model_dow = Model_Keras(model_dow_in, model_dow_out)

        # model_promo = Sequential()
        # model_promo.add(Dense(1, input_dim=1))
        # models.append(model_promo)
        # # models.append(model_promo.output)

        model_promo_in = Input(shape=(1,))
        models_in.append(model_promo_in)
        model_promo_out = Dense(1, input_dim=1)(model_promo_in)
        models_out.append(model_promo_out)
        model_promo = Model_Keras(model_promo_in, model_promo_out)

        # model_year = Sequential()
        # model_year.add(Embedding(3, 2, input_length=1))
        # # model_year.add(Reshape(dims=(2,)))
        # model_year.add(Reshape((2,)))
        # models.append(model_year)
        # # models.append(model_year.output)

        model_year_in = Input(shape=(1,))
        models_in.append(model_year_in)
        model_year_out = Embedding(3, 2, input_length=1)(model_year_in)
        model_year_out = Reshape(target_shape=(2,))(model_year_out)
        models_out.append(model_year_out)
        model_year = Model_Keras(model_year_in, model_year_out)

        # model_month = Sequential()
        # model_month.add(Embedding(12, 6, input_length=1))
        # # model_month.add(Reshape(dims=(6,)))
        # model_month.add(Reshape((6,)))
        # models.append(model_month)
        # # models.append(model_month.output)

        model_month_in = Input(shape=(1,))
        models_in.append(model_month_in)
        model_month_out = Embedding(12, 6, input_length=1)(model_month_in)
        model_month_out = Reshape(target_shape=(6,))(model_month_out)
        models_out.append(model_month_out)
        model_month = Model_Keras(model_month_in, model_month_out)

        # model_day = Sequential()
        # model_day.add(Embedding(31, 10, input_length=1))
        # # model_day.add(Reshape(dims=(10,)))
        # model_day.add(Reshape((10,)))
        # models.append(model_day)
        # # models.append(model_day.output)

        model_day_in = Input(shape=(1,))
        models_in.append(model_day_in)
        model_day_out = Embedding(31, 10, input_length=1)(model_day_in)
        model_day_out = Reshape(target_shape=(10,))(model_day_out)
        models_out.append(model_day_out)
        model_day = Model_Keras(model_day_in, model_day_out)

        # model_stateholiday = Sequential()
        # model_stateholiday.add(Embedding(4, 3, input_length=1))
        # # model_stateholiday.add(Reshape(dims=(3,)))
        # model_stateholiday.add(Reshape((3,)))
        # models.append(model_stateholiday)
        # # models.append(model_stateholiday.output)

        model_stateholiday_in = Input(shape=(1,))
        models_in.append(model_stateholiday_in)
        model_stateholiday_out = Embedding(4, 3, input_length=1)(model_stateholiday_in)
        model_stateholiday_out = Reshape(target_shape=(3,))(model_stateholiday_out)
        models_out.append(model_stateholiday_out)
        model_stateholiday = Model_Keras(model_stateholiday_in, model_stateholiday_out)

        # model_school = Sequential()
        # model_school.add(Dense(1, input_dim=1))
        # models.append(model_school)
        # # models.append(model_school.output)

        model_school_in = Input(shape=(1,))
        models_in.append(model_school_in)
        model_school_out = Dense(1, input_dim=1)(model_school_in)
        models_out.append(model_school_out)
        model_school = Model_Keras(model_school_in, model_school_out)

        # model_competemonths = Sequential()
        # model_competemonths.add(Embedding(25, 2, input_length=1))
        # # model_competemonths.add(Reshape(dims=(2,)))
        # model_competemonths.add(Reshape((2,)))
        # models.append(model_competemonths)
        # # models.append(model_competemonths.output)

        model_competemonths_in = Input(shape=(1,))
        models_in.append(model_competemonths_in)
        model_competemonths_out = Embedding(25, 2, input_length=1)(model_competemonths_in)
        model_competemonths_out = Reshape(target_shape=(2,))(model_competemonths_out)
        models_out.append(model_competemonths_out)
        model_competemonths = Model_Keras(model_competemonths_in, model_competemonths_out)

        # model_promo2weeks = Sequential()
        # model_promo2weeks.add(Embedding(26, 1, input_length=1))
        # # model_promo2weeks.add(Reshape(dims=(1,)))
        # model_promo2weeks.add(Reshape((1,)))
        # models.append(model_promo2weeks)
        # # models.append(model_promo2weeks.output)

        model_promo2weeks_in = Input(shape=(1,))
        models_in.append(model_promo2weeks_in)
        model_promo2weeks_out = Embedding(26, 1, input_length=1)(model_promo2weeks_in)
        model_promo2weeks_out = Reshape(target_shape=(1,))(model_promo2weeks_out)
        models_out.append(model_promo2weeks_out)
        model_promo2weeks = Model_Keras(model_promo2weeks_in, model_promo2weeks_out)

        # model_lastestpromo2months = Sequential()
        # model_lastestpromo2months.add(Embedding(4, 1, input_length=1))
        # # model_lastestpromo2months.add(Reshape(dims=(1,)))
        # model_lastestpromo2months.add(Reshape((1,)))
        # models.append(model_lastestpromo2months)
        # # models.append(model_lastestpromo2months.output)

        model_lastestpromo2months_in = Input(shape=(1,))
        models_in.append(model_lastestpromo2months_in)
        model_lastestpromo2months_out = Embedding(4, 1, input_length=1)(model_lastestpromo2months_in)
        model_lastestpromo2months_out = Reshape(target_shape=(1,))(model_lastestpromo2months_out)
        models_out.append(model_lastestpromo2months_out)
        model_lastestpromo2months = Model_Keras(model_lastestpromo2months_in, model_lastestpromo2months_out)

        # model_distance = Sequential()
        # model_distance.add(Dense(1, input_dim=1))
        # models.append(model_distance)
        # # models.append(model_distance.output)

        model_distance_in = Input(shape=(1,))
        models_in.append(model_distance_in)
        model_distance_out = Dense(1, input_dim=1)(model_distance_in)
        models_out.append(model_distance_out)
        model_distance = Model_Keras(model_distance_in, model_distance_out)

        # model_storetype = Sequential()
        # model_storetype.add(Embedding(5, 2, input_length=1))
        # # model_storetype.add(Reshape(dims=(2,)))
        # model_storetype.add(Reshape((2,)))
        # models.append(model_storetype)
        # # models.append(model_storetype.output)

        model_storetype_in = Input(shape=(1,))
        models_in.append(model_storetype_in)
        model_storetype_out = Embedding(5, 2, input_length=1)(model_storetype_in)
        model_storetype_out = Reshape(target_shape=(2,))(model_storetype_out)
        models_out.append(model_storetype_out)
        model_storetype = Model_Keras(model_storetype_in, model_storetype_out)

        # model_assortment = Sequential()
        # model_assortment.add(Embedding(4, 3, input_length=1))
        # # model_assortment.add(Reshape(dims=(3,)))
        # model_assortment.add(Reshape((3,)))
        # models.append(model_assortment)
        # # models.append(model_assortment.output)

        model_assortment_in = Input(shape=(1,))
        models_in.append(model_assortment_in)
        model_assortment_out = Embedding(4, 3, input_length=1)(model_assortment_in)
        model_assortment_out = Reshape(target_shape=(3,))(model_assortment_out)
        models_out.append(model_assortment_out)
        model_assortment = Model_Keras(model_assortment_in, model_assortment_out)

        # model_promointerval = Sequential()
        # model_promointerval.add(Embedding(4, 3, input_length=1))
        # # model_promointerval.add(Reshape(dims=(3,)))
        # model_promointerval.add(Reshape((3,)))
        # models.append(model_promointerval)
        # # models.append(model_promointerval.output)

        model_promointerval_in = Input(shape=(1,))
        models_in.append(model_promointerval_in)
        model_promointerval_out = Embedding(4, 3, input_length=1)(model_promointerval_in)
        model_promointerval_out = Reshape(target_shape=(3,))(model_promointerval_out)
        models_out.append(model_promointerval_out)
        model_promointerval = Model_Keras(model_promointerval_in, model_promointerval_out)

        # model_competyear = Sequential()
        # model_competyear.add(Embedding(18, 4, input_length=1))
        # # model_competyear.add(Reshape(dims=(4,)))
        # model_competyear.add(Reshape((4,)))
        # models.append(model_competyear)
        # # models.append(model_competyear.output)

        model_competyear_in = Input(shape=(1,))
        models_in.append(model_competyear_in)
        model_competyear_out = Embedding(18, 4, input_length=1)(model_competyear_in)
        model_competyear_out = Reshape(target_shape=(4,))(model_competyear_out)
        models_out.append(model_competyear_out)
        model_competyear = Model_Keras(model_competyear_in, model_competyear_out)

        # model_promotyear = Sequential()
        # model_promotyear.add(Embedding(8, 4, input_length=1))
        # # model_promotyear.add(Reshape(dims=(4,)))
        # model_promotyear.add(Reshape((4,)))
        # models.append(model_promotyear)
        # # models.append(model_promotyear.output)

        model_promotyear_in = Input(shape=(1,))
        models_in.append(model_promotyear_in)
        model_promotyear_out = Embedding(8, 4, input_length=1)(model_promotyear_in)
        model_promotyear_out = Reshape(target_shape=(4,))(model_promotyear_out)
        models_out.append(model_promotyear_out)
        model_promotyear = Model_Keras(model_promotyear_in, model_promotyear_out)

        # model_germanstate = Sequential()
        # model_germanstate.add(Embedding(12, 6, input_length=1))
        # # model_germanstate.add(Reshape(dims=(6,)))
        # model_germanstate.add(Reshape((6,)))
        # models.append(model_germanstate)
        # # models.append(model_germanstate.output)

        model_germanstate_in = Input(shape=(1,))
        models_in.append(model_germanstate_in)
        model_germanstate_out = Embedding(12, 6, input_length=1)(model_germanstate_in)
        model_germanstate_out = Reshape(target_shape=(6,))(model_germanstate_out)
        models_out.append(model_germanstate_out)
        model_germanstate = Model_Keras(model_germanstate_in, model_germanstate_out)

        # model_woy = Sequential()
        # model_woy.add(Embedding(53, 2, input_length=1))
        # # model_woy.add(Reshape(dims=(2,)))
        # model_woy.add(Reshape((2,)))
        # models.append(model_woy)
        # # models.append(model_woy.output)

        model_woy_in = Input(shape=(1,))
        models_in.append(model_woy_in)
        model_woy_out = Embedding(53, 2, input_length=1)(model_woy_in)
        model_woy_out = Reshape(target_shape=(2,))(model_woy_out)
        models_out.append(model_woy_out)
        model_woy = Model_Keras(model_woy_in, model_woy_out)

        # model_temperature = Sequential()
        # model_temperature.add(Dense(3, input_dim=3))
        # models.append(model_temperature)
        # # models.append(model_temperature.output)

        model_temperature_in = Input(shape=(1,))
        models_in.append(model_temperature_in)
        model_temperature_out = Dense(3, input_dim=3)(model_temperature_in)
        models_out.append(model_temperature_out)
        model_temperature = Model_Keras(model_temperature_in, model_temperature_out)

        # model_humidity = Sequential()
        # model_humidity.add(Dense(3, input_dim=3))
        # models.append(model_humidity)
        # # models.append(model_humidity.output)

        model_humidity_in = Input(shape=(1,))
        models_in.append(model_humidity_in)
        model_humidity_out = Dense(3, input_dim=3)(model_humidity_in)
        models_out.append(model_humidity_out)
        model_humidity = Model_Keras(model_humidity_in, model_humidity_out)

        # model_wind = Sequential()
        # model_wind.add(Dense(2, input_dim=2))
        # models.append(model_wind)
        # # models.append(model_wind.output)

        model_wind_in = Input(shape=(1,))
        models_in.append(model_wind_in)
        model_wind_out = Dense(2, input_dim=2)(model_wind_in)
        models_out.append(model_wind_out)
        model_wind = Model_Keras(model_wind_in, model_wind_out)

        # model_cloud = Sequential()
        # model_cloud.add(Dense(1, input_dim=1))
        # models.append(model_cloud)
        # # models.append(model_cloud.output)

        model_cloud_in = Input(shape=(1,))
        models_in.append(model_cloud_in)
        model_cloud_out = Dense(1, input_dim=1)(model_cloud_in)
        models_out.append(model_cloud_out)
        model_cloud = Model_Keras(model_cloud_in, model_cloud_out)

        # model_weatherevent = Sequential()
        # model_weatherevent.add(Embedding(22, 4, input_length=1))
        # # model_weatherevent.add(Reshape(dims=(4,)))
        # model_weatherevent.add(Reshape((4,)))
        # models.append(model_weatherevent)
        # # models.append(model_weatherevent.output)

        model_weatherevent_in = Input(shape=(1,))
        models_in.append(model_weatherevent_in)
        model_weatherevent_out = Embedding(22, 4, input_length=1)(model_weatherevent_in)
        model_weatherevent_out = Reshape(target_shape=(4,))(model_weatherevent_out)
        models_out.append(model_weatherevent_out)
        model_weatherevent = Model_Keras(model_weatherevent_in, model_weatherevent_out)

        # model_promo_forward = Sequential()
        # model_promo_forward.add(Embedding(8, 1, input_length=1))
        # # model_promo_forward.add(Reshape(dims=(1,)))
        # model_promo_forward.add(Reshape((1,)))
        # models.append(model_promo_forward)
        # # models.append(model_promo_forward.output)

        model_promo_forward_in = Input(shape=(1,))
        models_in.append(model_promo_forward_in)
        model_promo_forward_out = Embedding(8, 1, input_length=1)(model_promo_forward_in)
        model_promo_forward_out = Reshape(target_shape=(1,))(model_promo_forward_out)
        models_out.append(model_promo_forward_out)
        model_promo_forward = Model_Keras(model_promo_forward_in, model_promo_forward_out)

        # model_promo_backward = Sequential()
        # model_promo_backward.add(Embedding(8, 1, input_length=1))
        # # model_promo_backward.add(Reshape(dims=(1,)))
        # model_promo_backward.add(Reshape((1,)))
        # models.append(model_promo_backward)
        # # models.append(model_promo_backward.output)

        model_promo_backward_in = Input(shape=(1,))
        models_in.append(model_promo_backward_in)
        model_promo_backward_out = Embedding(8, 1, input_length=1)(model_promo_backward_in)
        model_promo_backward_out = Reshape(target_shape=(1,))(model_promo_backward_out)
        models_out.append(model_promo_backward_out)
        model_promo_backward = Model_Keras(model_promo_backward_in, model_promo_backward_out)

        # model_stateholiday_forward = Sequential()
        # model_stateholiday_forward.add(Embedding(8, 1, input_length=1))
        # # model_stateholiday_forward.add(Reshape(dims=(1,)))
        # model_stateholiday_forward.add(Reshape((1,)))
        # models.append(model_stateholiday_forward)
        # # models.append(model_stateholiday_forward.output)

        model_stateholiday_forward_in = Input(shape=(1,))
        models_in.append(model_stateholiday_forward_in)
        model_stateholiday_forward_out = Embedding(8, 1, input_length=1)(model_stateholiday_forward_in)
        model_stateholiday_forward_out = Reshape(target_shape=(1,))(model_stateholiday_forward_out)
        models_out.append(model_stateholiday_forward_out)
        model_stateholiday_forward = Model_Keras(model_stateholiday_forward_in, model_stateholiday_forward_out)

        # model_sateholiday_backward = Sequential()
        # model_sateholiday_backward.add(Embedding(8, 1, input_length=1))
        # # model_sateholiday_backward.add(Reshape(dims=(1,)))
        # model_sateholiday_backward.add(Reshape((1,)))
        # models.append(model_sateholiday_backward)
        # # models.append(model_sateholiday_backward.output)

        model_sateholiday_backward_in = Input(shape=(1,))
        models_in.append(model_sateholiday_backward_in)
        model_sateholiday_backward_out = Embedding(8, 1, input_length=1)(model_sateholiday_backward_in)
        model_sateholiday_backward_out = Reshape(target_shape=(1,))(model_sateholiday_backward_out)
        models_out.append(model_sateholiday_backward_out)
        model_sateholiday_backward = Model_Keras(model_sateholiday_backward_in, model_sateholiday_backward_out)

        # model_stateholiday_count_forward = Sequential()
        # model_stateholiday_count_forward.add(Embedding(3, 1, input_length=1))
        # # model_stateholiday_count_forward.add(Reshape(dims=(1,)))
        # model_stateholiday_count_forward.add(Reshape((1,)))
        # models.append(model_stateholiday_count_forward)
        # # models.append(model_stateholiday_count_forward.output)

        model_stateholiday_count_forward_in = Input(shape=(1,))
        models_in.append(model_stateholiday_count_forward_in)
        model_stateholiday_count_forward_out = Embedding(3, 1, input_length=1)(model_stateholiday_count_forward_in)
        model_stateholiday_count_forward_out = Reshape(target_shape=(1,))(model_stateholiday_count_forward_out)
        models_out.append(model_stateholiday_count_forward_out)
        model_stateholiday_count_forward = Model_Keras(model_stateholiday_count_forward_in, model_stateholiday_count_forward_out)

        # model_stateholiday_count_backward = Sequential()
        # model_stateholiday_count_backward.add(Embedding(3, 1, input_length=1))
        # # model_stateholiday_count_backward.add(Reshape(dims=(1,)))
        # model_stateholiday_count_backward.add(Reshape((1,)))
        # models.append(model_stateholiday_count_backward)
        # # models.append(model_stateholiday_count_backward.output)

        model_stateholiday_count_backward_in = Input(shape=(1,))
        models_in.append(model_stateholiday_count_backward_in)
        model_stateholiday_count_backward_out = Embedding(3, 1, input_length=1)(model_stateholiday_count_backward_in)
        model_stateholiday_count_backward_out = Reshape(target_shape=(1,))(model_stateholiday_count_backward_out)
        models_out.append(model_stateholiday_count_backward_out)
        model_stateholiday_count_backward = Model_Keras(model_stateholiday_count_backward_in, model_stateholiday_count_backward_out)

        # model_schoolholiday_forward = Sequential()
        # model_schoolholiday_forward.add(Embedding(8, 1, input_length=1))
        # # model_schoolholiday_forward.add(Reshape(dims=(1,)))
        # model_schoolholiday_forward.add(Reshape((1,)))
        # models.append(model_schoolholiday_forward)
        # # models.append(model_schoolholiday_forward.output)

        model_schoolholiday_forward_in = Input(shape=(1,))
        models_in.append(model_schoolholiday_forward_in)
        model_schoolholiday_forward_out = Embedding(8, 1, input_length=1)(model_schoolholiday_forward_in)
        model_schoolholiday_forward_out = Reshape(target_shape=(1,))(model_schoolholiday_forward_out)
        models_out.append(model_schoolholiday_forward_out)
        model_schoolholiday_forward = Model_Keras(model_schoolholiday_forward_in, model_schoolholiday_forward_out)

        # model_schoolholiday_backward = Sequential()
        # model_schoolholiday_backward.add(Embedding(8, 1, input_length=1))
        # # model_schoolholiday_backward.add(Reshape(dims=(1,)))
        # model_schoolholiday_backward.add(Reshape((1,)))
        # models.append(model_schoolholiday_backward)
        # # models.append(model_schoolholiday_backward.output)

        model_schoolholiday_backward_in = Input(shape=(1,))
        models_in.append(model_schoolholiday_backward_in)
        model_schoolholiday_backward_out = Embedding(8, 1, input_length=1)(model_schoolholiday_backward_in)
        model_schoolholiday_backward_out = Reshape(target_shape=(1,))(model_schoolholiday_backward_out)
        models_out.append(model_schoolholiday_backward_out)
        model_schoolholiday_backward = Model_Keras(model_schoolholiday_backward_in, model_schoolholiday_backward_out)

        # model_googletrend_de = Sequential()
        # model_googletrend_de.add(Dense(1, input_dim=1))
        # models.append(model_googletrend_de)
        # # models.append(model_googletrend_de.output)

        model_googletrend_de_in = Input(shape=(1,))
        models_in.append(model_googletrend_de_in)
        model_googletrend_de_out = Dense(1, input_dim=1)(model_googletrend_de_in)
        models_out.append(model_googletrend_de_out)
        model_googletrend_de = Model_Keras(model_googletrend_de_in, model_googletrend_de_out)

        # model_googletrend_state = Sequential()
        # model_googletrend_state.add(Dense(1, input_dim=1))
        # models.append(model_googletrend_state)
        # # models.append(model_googletrend_state.output)

        model_googletrend_state_in = Input(shape=(1,))
        models_in.append(model_googletrend_state_in)
        model_googletrend_state_out = Dense(1, input_dim=1)(model_googletrend_state_in)
        models_out.append(model_googletrend_state_out)
        model_googletrend_state = Model_Keras(model_googletrend_state_in, model_googletrend_state_out)

        # model_weather = Sequential()
        # model_weather.add(Merge([model_temperature, model_humidity, model_wind, model_weatherevent], mode='concat'))
        # model_weather.add(Dense(1))
        # model_weather.add(Activation('relu'))
        # models.append(model_weather)

        # self.model = Sequential()
        # self.model.add(Merge(models, mode='concat'))
        # self.model.add(Dropout(0.02))
        # self.model.add(Dense(1000, init='uniform'))
        # self.model.add(Activation('relu'))
        # self.model.add(Dense(500, init='uniform'))
        # self.model.add(Activation('relu'))
        # self.model.add(Dense(1))
        # self.model.add(Activation('sigmoid'))

        concatenated = concatenate(models_out)
        output = Dropout(0.02)(concatenated)
        output = Dense(1000, init='uniform')(output)
        output = Activation('relu')(output)
        output = Dense(500, init='uniform')(output)
        output = Activation('relu')(output)
        output = Dense(1)(output)
        output = Activation('sigmoid')(output)

        self.model = Model_Keras(models_in, output)

        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        return numpy.exp(val * self.max_log_y)

    def fit(self):
        if self.train_ratio < 1:
            self.model.fit(self.preprocessing(self.X), self._val_for_fit(self.y),
                           validation_data=(self.preprocessing(self.X_val), self._val_for_fit(self.y_val)),
                           nb_epoch=self.nb_epoch, batch_size=128,
                           # callbacks=[self.checkpointer],
                           )
            # self.model.load_weights('best_model_weights.hdf5')
            print("Result on validation data: ", self.evaluate())
        else:
            self.model.fit(self.preprocessing(self.X), self._val_for_fit(self.y),
                           nb_epoch=self.nb_epoch, batch_size=128)

    def guess(self, feature):
        feature = numpy.array(feature).reshape(1, -1)
        return self._val_for_pred(self.model.predict(self.preprocessing(feature)))[0][0]
