from DeepLearningTools.ReturnGenerators import return_generators, plot_scroll_Image
import tensorflow as tf
import os
from DeepLearningTools.ReturnModels import GeneratorBMA2
import pandas as pd


def main():
    base_path = r'\\ad.ucsd.edu\ahs\radon\research\Bojechko'
    df = pd.read_excel(os.path.join(base_path, 'Model_Outputs', 'Model_Parameters.xlsx')).to_dict()
    for index in df['Model_ID'].keys():
        model_id = df['Model_ID'][index]
        out_path = os.path.join(base_path, 'Model_Outputs', f'Model_{model_id}')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        else:
            continue
        kwargs = {}
        for key in df.keys():
            kwargs[key] = df[key][index]
        train_gen, valid_gen = return_generators(base_path=os.path.join(base_path, 'TFRecords',
                                                                        'TrainNoNormalizationMultipleProj'),
                                                 **kwargs)
        mae = tf.keras.metrics.MeanAbsoluteError()
        loss = tf.keras.losses.MeanSquaredError()

        model = GeneratorBMA2(**kwargs)
        cos_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=kwargs['lr'],
                                                                      first_decay_steps=int(len(train_gen) * 2),
                                                                      t_mul=2.0, m_mul=1.0, alpha=0.0)
        model.compile(optimizer=tf.keras.optimizers.Adam(cos_decay), loss=loss, metrics=[mae])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(out_path, 'checkpoint/cp.ckpt'),
                                                        verbose=1, save_best_only=True, save_weights_only=True)
        print(out_path)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=out_path, profile_batch=0, write_graph=False)
        model.fit(train_gen.repeat(), epochs=2045, steps_per_epoch=len(train_gen)*5,
                  validation_data=valid_gen, validation_freq=1, callbacks=[tensorboard, checkpoint])
        model.save_weights(os.path.join(out_path, 'final_model_weights.h5'))


if __name__ == '__main__':
    main()
