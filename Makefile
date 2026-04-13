

local-model:
	python inc/jetmoe_install.py
	sed "s/\(import scattermoe\s*\)/#\1/" inc/jetmoe-local/modeling_jetmoe.py
	sed "s/\(self.input_linear = scattermoe.parallel_experts.ParallelExperts(num_experts, input_size, hidden_size * 2 if glu else hidden_size)\)\
        \(self.output_linear = scattermoe.parallel_experts.ParallelExperts(num_experts, hidden_size, input_size)\)/#\1\
		#\2\
		input_out_dim = hidden_size * 2 if glu else hidden_size\
        \
        self.input_linear = nn.ModuleList([\
            nn.Linear(input_size, input_out_dim, bias=False) \
            for _ in range(num_experts)\
        ])\
        \
        self.output_linear = nn.ModuleList([\
            nn.Linear(hidden_size, input_size, bias=False) \
            for _ in range(num_experts)\
        ])/"

clean:
	rm -rf inc/jetmoe-local
