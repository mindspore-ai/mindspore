- **weight_decay** - 可选。如果键中存在"weight_decay”，则使用对应的值作为权重衰减值。如果没有，则使用优化器中配置的 `weight_decay` 作为权重衰减值。
  值得注意的是， `weight_decay` 可以是常量，也可以是Cell类型。Cell类型的weight decay用于实现动态weight decay算法。动态权重衰减和动态学习率相似，
  用户需要自定义一个输入为global step的weight_decay_schedule。在训练的过程中，优化器会调用WeightDecaySchedule的实例来获取当前step的weight decay值。
