1. 原本的寫法 
    new_saver = tf.train.import_meta_graph('itrichess_new.ckpt.meta')    
    new_saver.restore(session, './itrichess_new.ckpt')


2. 我改為
    new_saver = tf.train.Saver()
    new_saver.restore(session, './itrichess_new.ckpt')

   不過這樣就沒有讀取到itrichess_new.ckpt.meta了 @@

   